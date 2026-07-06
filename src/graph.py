import os
import sys
import re
import asyncio
import json
import datetime
from typing import TypedDict, List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# MCP SDK Client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import helpers from utils
from src.utils import (
    get_llm,
    get_content,
    convert_usd_to_inr_in_text,
    enrich_context_with_web_searches,
    calculate_distance,
    resolve_coordinates_via_llm
)

# -------------------------------------------------------------
# 1. State Definition
# -------------------------------------------------------------
class AgentState(TypedDict):
    """
    TypedDict state representing the data passed between nodes in the LangGraph compilation journey.
    """
    user_request: str
    max_budget: float
    travel_month: str
    draft_itinerary: str
    validation_errors: List[str]
    budget_errors: List[str]
    iteration_count: int
    final_itinerary_markdown: str
    final_itinerary_json: str


# Helper to run async code inside sync LangGraph nodes (necessary for Streamlit/asyncio compatibility)
def run_async(coro):
    """
    Executes an asynchronous coroutine safely, resolving event loop conflicts
    that commonly occur when nesting asyncio in Streamlit/LangGraph.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.new_event_loop().run_until_complete(coro))
            return future.result()
    else:
        return loop.run_until_complete(coro)


# -------------------------------------------------------------
# 2. Pydantic Schemas for Structured LLM Outputs
# -------------------------------------------------------------
class LandmarkVisit(BaseModel):
    landmark: str = Field(description="Name of the landmark or location visited (e.g. 'Tokyo National Museum', 'Kyoto Imperial Palace')")
    day_number: int = Field(description="The day number of the trip (1, 2, etc.)")
    day_of_week: str = Field(description="The day of the week (e.g. Monday, Tuesday, Sunday) for this day")

class ItineraryAnalysis(BaseModel):
    visits: List[LandmarkVisit] = Field(description="List of all landmark visits extracted from the itinerary draft")


# Async helper to call the standalone MCP server via stdio
async def call_mcp_server(location: str) -> str:
    """
    Starts the standalone MCP server as a sub-process and queries the check_local_constraints tool.
    """
    server_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_server.py"))
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script]
    )
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Call check_local_constraints tool on MCP server
                result = await session.call_tool("check_local_constraints", arguments={"location": location})
                if result and result.content:
                    return result.content[0].text
                return f"No constraints found for {location}"
    except Exception as e:
        return f"Error executing MCP Server query: {str(e)}"


# -------------------------------------------------------------
# 3. Multi-Agent Nodes Implementation
# -------------------------------------------------------------

# Node 1: Security Scrubber Middleware
def security_scrubber_node(state: AgentState) -> Dict[str, Any]:
    """
    Security Feature: Intercepts raw user inputs and redacts sensitive PII
    (Names, Emails, and Phone Numbers) using robust regex & rule-based scrubbing.
    Also performs out-of-scope validation to ensure requests relate to travel.
    """
    raw_prompt = state["user_request"]
    
    # 1. Scope Guard Check
    llm = get_llm("structured")
    scope_prompt = f"""
You are the Scope Gatekeeper for a Personalized Travel & Holiday Planning Agent.
Evaluate if the following user request is related to travel, geography, lodging, transportation, itineraries, booking, sightseeing, landmarks, or holiday planning.

User Request: "{raw_prompt}"

Output format: Output ONLY a valid JSON object with a single key "in_scope" (boolean). True if the request is related to travel/planning, False otherwise.
Do not add introductory or concluding remarks, or markdown code block syntax.
Example:
{{"in_scope": true}}
"""
    try:
        response = llm.invoke(scope_prompt)
        text = get_content(response).strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        in_scope = bool(data.get("in_scope", True))
    except Exception:
        in_scope = True
        
    if not in_scope:
        return {
            "user_request": "__OUT_OF_SCOPE__"
        }
        
    # 2. Email Redaction
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    scrubbed = re.sub(email_pattern, "[REDACTED_EMAIL]", raw_prompt)
    
    # 3. Phone Number Redaction
    phone_pattern = r'\+?\d{1,4}[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    scrubbed = re.sub(phone_pattern, "[REDACTED_PHONE]", scrubbed)
    
    # 4. Name Redaction (handles phrases like 'My name is X', 'I am X', and specific names)
    name_phrases = [
        (r'(my name is|i am|call me|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', r'\1 [REDACTED_NAME]'),
        (r'\b(sahil|sarah|john|smith|doe)\b', r'[REDACTED_NAME]')
    ]
    for pattern, replacement in name_phrases:
        scrubbed = re.sub(pattern, replacement, scrubbed, flags=re.IGNORECASE)
        
    return {
        "user_request": scrubbed
    }


# Node 2: Planner Agent
def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Planner Persona: Drafts the day-by-day itinerary or actively modifies
    the previous draft to resolve validation/budget errors or apply user changes.
    Queries DuckDuckGo search to retrieve real prices and scheduling constraints.
    """
    if state["user_request"] == "__OUT_OF_SCOPE__":
        return {
            "draft_itinerary": "I am sorry, but that request is out of my scope. As a Personalized Holiday Planning Agent, I can only assist with travel itineraries, destination mapping, transport/hotel costing, and holiday planning."
        }

    llm = get_llm("default")
    
    # 1. Fetch real-time web search context
    search_context = enrich_context_with_web_searches(state["user_request"], state.get("draft_itinerary", ""))
    today_str = datetime.date.today().strftime("%B %d, %Y (%A)")
    
    # Check if this is a revision loop or a fresh draft
    has_previous_draft = bool(state.get("draft_itinerary"))
    if has_previous_draft:
        prompt = f"""
You are the Lead Travel Planner Agent. Your job is to REVISE the previous travel itinerary based on the user's new request/feedback, real-time web searches, and any validation/budget errors.

Current Calendar Date: {today_str}
Travel Dates: {state['travel_month']}
Max Budget: ₹{state['max_budget']}
User's Revision Request/Feedback: {state['user_request']}

---
REAL-TIME WEB SEARCH RESULTS (Use this to get accurate transport fares, hotel prices, and attraction schedules/closed days):
{search_context}
---

---
PREVIOUS DRAFT ITINERARY:
{state['draft_itinerary']}
---

CRITICAL FEEDBACK TO ADDRESS:
1. Apply the user's revision request: "{state['user_request']}"
2. Resolve any validation or budget errors below:
- Validation Errors: {state.get('validation_errors', [])}
- Budget Errors: {state.get('budget_errors', [])}

CRITICAL TRANSPORTATION RULE:
- For transportation between the user's starting point and their destination (or between cities), you MUST explicitly list and compare multiple travel options (e.g. train routes, highway buses, driving routes, and flights where applicable) using the real-time search prices above, with their estimated travel times and costs in Indian Rupees (₹).
- Do not just output a single flight option. Provide a clear 'Transportation Options' comparison section.

CRITICAL PRICE/SCHEDULE ACCURACY RULES:
- Use the real-time search results above to estimate realistic ticket fares, hotel prices per night, and dining/food costs in Indian Rupees (₹). Avoid hardcoding fake or generic costs.
- Do not schedule visits to attractions on days they are closed (refer to the landmark info above; for example, Ajanta Caves is closed to visitors every Monday). Double check the days of the week for your travel dates.

CRITICAL CURRENCY RULES:
- DO NOT USE THE '$' SIGN OR THE DOLLAR CURRENCY.
- YOU MUST DENOMINATE ALL COSTS AND EXPENSES SOLELY IN INDIAN RUPEES (₹).
- YOU MUST REPRESENT PRICES AND COSTS AS ESTIMATED RANGES, FOLLOWED BY THE UPPER BOUND IN BRACKETS. Format: "Estimated range: ₹Min - ₹Max [Cost: ₹Max]" (e.g. "Estimated range: ₹500 - ₹600 [Cost: ₹600]", "Estimated range: ₹10,000 - ₹12,000 [Cost: ₹12000]"). This is required for both traveler friendliness and budget tracking.

GUIDELINES FOR OUTPUT:
- Keep the day-by-day structured itinerary format.
- Output ONLY the updated markdown itinerary. Do not include introductory conversational text.
"""
    else:
        prompt = f"""
You are the Lead Travel Planner Agent. Your job is to draft a comprehensive, day-by-day travel itinerary matching the user request, enriched with real-time web search costs and schedule checks.

Current Calendar Date: {today_str}
Travel Dates: {state['travel_month']}
Max Budget: ₹{state['max_budget']}
User Request: {state['user_request']}

---
REAL-TIME WEB SEARCH RESULTS (Use this to get accurate transport fares, hotel prices, and attraction schedules/closed days):
{search_context}
---

GUIDELINES & RULES:
1. Create a detailed day-by-day travel plan (e.g. Day 1, Day 2, etc.) specifying which days are which days of the week starting with Monday (e.g. "Day 1 (Monday)"). Double check the days of the week for your travel dates.
2. Explicitly label each landmark visited.
3. Use the real-time search results above to estimate realistic ticket fares, hotel prices per night, and dining/food costs in Indian Rupees (₹). Avoid hardcoding fake or generic costs.
4. Do not schedule visits to attractions on days they are closed (refer to the landmark info above; for example, Ajanta Caves is closed to visitors every Monday).
5. For EVERY activity, hotel night, and transit route, you must represent the estimated price as a range followed by the upper bound inside brackets: "Estimated range: ₹Min - ₹Max [Cost: ₹Max]" (e.g., "Estimated range: ₹500 - ₹600 [Cost: ₹600]", "Estimated range: ₹10,000 - ₹12,000 [Cost: ₹12000]", "Estimated range: ₹0 [Cost: ₹0]"). This format is required for both user readability and budget tracking.
6. CRITICAL CURRENCY RULE: Under no circumstances should you use the Dollar symbol ($) or output costs in Dollars. You must convert all expenses to Indian Rupees (₹) and annotate them using the ₹ symbol.
7. CRITICAL TRANSPORTATION RULE: For transit between the user's home/start city and the destination (or between cities), you MUST explicitly outline and compare multiple travel options (such as train, bus, driving, and flight details) with travel times and real-time search Rupees costs. Do not default or jump only to flights. Present these choices clearly under a 'Transportation Options' heading.
8. Keep the total cost within the maximum budget: ₹{state['max_budget']}.
9. Do not plan visits to destinations in different cities on the exact same day unless travel is accounted for.

Output ONLY the markdown itinerary. Do not add introductory or concluding chat remarks.
"""
    
    response = llm.invoke(prompt)
    cleaned_content = convert_usd_to_inr_in_text(get_content(response))
    return {
        "draft_itinerary": cleaned_content
    }


# Node 3: Validator Agent (MCP & Geodesic Distance Skills)
def validator_node(state: AgentState) -> Dict[str, Any]:
    """
    Validator Persona: Parses the itinerary draft, queries the local MCP server for operational
    and closure constraints, and calls the geodesic distance skill to detect travel overlaps.
    """
    if state.get("user_request") == "__OUT_OF_SCOPE__":
        return {
            "validation_errors": []
        }

    draft = state["draft_itinerary"]
    validation_errors = []
    
    # 1. Parse itinerary landmarks and scheduling using Structured Output LLM (with JSON fallback)
    today_str = datetime.date.today().strftime("%B %d, %Y (%A)")
    dates_str = state.get("travel_month", "not specified")
    extraction_prompt = f"""
Current Calendar Date: {today_str}
Travel Dates: {dates_str}

Parse the following travel itinerary and extract all landmark/location visits.
For each visit, determine:
1. The exact landmark name (e.g., 'Tokyo National Museum', 'Kyoto Imperial Palace', 'Mount Fuji', 'Ghibli Museum').
2. The day number of the trip (1, 2, 3, etc.).
3. The day of the week (e.g. Monday, Tuesday, Wednesday, etc.) for that day of the trip. Match the day numbers to their exact days of the week based on the travel dates: {dates_str}.

Itinerary:
{draft}
"""
    fallback_prompt = f"""
Current Calendar Date: {today_str}
Travel Dates: {dates_str}

Parse the following travel itinerary and extract all landmark/location visits.
You MUST output a valid JSON object matching this schema:
{{
  "visits": [
    {{
      "landmark": "Name of the landmark or location visited (e.g. 'Tokyo National Museum', 'Kyoto Imperial Palace')",
      "day_number": int,
      "day_of_week": "The day of the week (e.g. Monday, Tuesday, Sunday) for this day based on the travel dates: {dates_str}"
    }}
  ]
}}

Output ONLY the raw JSON block. Do not include markdown code block characters like ```json or any other conversational text.

Itinerary:
{draft}
"""
    try:
        llm = get_llm("structured")
        structured_llm = llm.with_structured_output(ItineraryAnalysis)
        analysis = structured_llm.invoke(extraction_prompt)
        visits = analysis.visits
    except Exception as e:
        print(f"[validator] structured_llm failed or not supported by provider ({e}). Falling back to JSON parsing...")
        try:
            llm = get_llm("structured")
            resp = llm.invoke(fallback_prompt)
            clean_json = get_content(resp).replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            visits = []
            for item in data.get("visits", []):
                visits.append(
                    LandmarkVisit(
                        landmark=item["landmark"],
                        day_number=int(item["day_number"]),
                        day_of_week=item["day_of_week"]
                    )
                )
        except Exception as err:
            print(f"[validator] JSON fallback parsing failed: {err}")
            visits = []
            
    # 2. Query Standalone MCP server for each landmark's temporal rules & constraints
    for visit in visits:
        landmark = visit.landmark
        mcp_result = run_async(call_mcp_server(landmark))
        
        # Check weekly closed day error
        closed_pattern = r'Closed Days:\s*([a-zA-Z\s,]+)'
        closed_match = re.search(closed_pattern, mcp_result, re.IGNORECASE)
        if closed_match:
            closed_days = [day.strip().lower() for day in closed_match.group(1).split(",")]
            visit_day = visit.day_of_week.strip().lower()
            if visit_day in closed_days:
                validation_errors.append(
                    f"Temporal Collision: Scheduled a visit to '{landmark}' on Day {visit.day_number} ({visit.day_of_week}), "
                    f"but it is closed on {visit.day_of_week}s. Detail: {mcp_result}"
                )
                
        # Check seasonal closed month error
        closed_months_pattern = r'Closed Months:\s*([a-zA-Z\s,]+)'
        closed_months_match = re.search(closed_months_pattern, mcp_result, re.IGNORECASE)
        if closed_months_match:
            closed_months = [m.strip().lower() for m in closed_months_match.group(1).split(",")]
            travel_month_lower = dates_str.lower()
            for m in closed_months:
                if m != "none" and m in travel_month_lower:
                    validation_errors.append(
                        f"Seasonal Collision: Scheduled a visit to '{landmark}' in {state['travel_month']}, "
                        f"but it is closed during {m.upper()} months. Detail: {mcp_result}"
                    )
                    
        # Check permit required step warning
        if "permit required" in mcp_result.lower():
            if "permit" not in draft.lower():
                validation_errors.append(
                    f"Permit Requirement: {landmark} (Day {visit.day_number}) requires an entry permit, "
                    f"but no permit acquisition step was found in the itinerary."
                )

    # 3. Geodesic Distance Collision Skill
    day_groups: Dict[int, List[LandmarkVisit]] = {}
    for visit in visits:
        day_groups.setdefault(visit.day_number, []).append(visit)
        
    for day_num, day_visits in day_groups.items():
        if len(day_visits) > 1:
            for i in range(len(day_visits)):
                for j in range(i + 1, len(day_visits)):
                    loc1 = day_visits[i].landmark
                    loc2 = day_visits[j].landmark
                    distance_msg = calculate_distance.invoke({"loc1": loc1, "loc2": loc2})
                    
                    if "NOT FEASIBLE" in distance_msg:
                        validation_errors.append(
                            f"Geographic Inconsistency: Scheduled both '{loc1}' and '{loc2}' on Day {day_num}. "
                            f"These locations are too far apart for the same day. Detail: {distance_msg}"
                        )
                        
    return {
        "validation_errors": validation_errors
    }


def get_main_location_via_llm(user_request: str) -> str:
    """
    Identifies the primary destination city/region from the user travel request.
    """
    try:
        llm = get_llm("structured")
        prompt = f"""
Identify the primary destination city or region from the following user travel request.
Output ONLY the name of the city/region (e.g. "Paris", "Aurangabad", "Tokyo"). Do not add any extra text or punctuation.

User Request: {user_request}
"""
        response = llm.invoke(prompt)
        return get_content(response).strip()
    except Exception:
        return "Tokyo"


# Node 4: Weather Adaptor Agent
def weather_adaptor_node(state: AgentState) -> Dict[str, Any]:
    """
    Weather Adaptor Persona: Evaluates travel month constraints against planned locations
    and appends specific errors and indoor alternative recommendations to validation_errors.
    Supports real-time weather forecasts if dates are within 14 days, otherwise estimates.
    """
    if state.get("user_request") == "__OUT_OF_SCOPE__":
        return {
            "validation_errors": []
        }

    travel_dates_str = state["travel_month"]
    draft = state["draft_itinerary"]
    validation_errors = list(state.get("validation_errors", []))
    
    # 1. Identify primary destination city/region
    destination = get_main_location_via_llm(state["user_request"])
    
    # 2. Default weather context
    today = datetime.date.today()
    weather_info = "Real-time forecast not available (dates are further in the future or not specified). Please estimate the typical climate/weather averages for this location during these dates."
    
    # Parse dates
    start_date, end_date = None, None
    if travel_dates_str:
        parts = travel_dates_str.split(" to ")
        if len(parts) == 2:
            def parse_single(d_str):
                for fmt in ('%b %d, %Y', '%Y-%m-%d', '%Y/%m/%d'):
                    try:
                        return datetime.datetime.strptime(d_str.strip(), fmt).date()
                    except ValueError:
                        continue
                return None
            start_date = parse_single(parts[0])
            end_date = parse_single(parts[1])
            
    if start_date and end_date:
        delta = start_date - today
        if 0 <= delta.days <= 14:
            coords = resolve_coordinates_via_llm(destination)
            if coords:
                lat, lon = coords
                url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,rain_sum,showers_sum,snowfall_sum&timezone=auto"
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        daily = data.get("daily", {})
                        time_list = daily.get("time", [])
                        temp_max = daily.get("temperature_2m_max", [])
                        temp_min = daily.get("temperature_2m_min", [])
                        rain_sum = daily.get("rain_sum", [])
                        
                        weather_reports = []
                        for i, t_str in enumerate(time_list):
                            t_date = datetime.datetime.strptime(t_str, "%Y-%m-%d").date()
                            if start_date <= t_date <= end_date:
                                rain_val = rain_sum[i] if i < len(rain_sum) else 0.0
                                weather_reports.append(
                                    f"- {t_date.strftime('%A, %b %d')}: Temp {temp_min[i]}°C to {temp_max[i]}°C, Rain: {rain_val}mm"
                                )
                        if weather_reports:
                            weather_info = f"Real-time Weather Forecast for {destination} from Open-Meteo API:\n" + "\n".join(weather_reports)
                except Exception as e:
                    print(f"Weather API failed: {e}")
                    
    # 3. Prompt LLM with weather context
    llm = get_llm("structured")
    prompt = f"""
You are the Weather Adaptor Agent. Your job is to check if the planned activities or landmarks in the itinerary conflict with the weather conditions or seasonal constraints at the destination.

Current Calendar Date: {today.strftime('%B %d, %Y (%A)')}
Destination: {destination}
Travel Dates: {travel_dates_str}

WEATHER INFORMATION & DATA:
{weather_info}

Review the itinerary:
{draft}

Seasonal/Weather Rules to check:
1. Outdoor vs Indoor viability: If heavy rain is forecast (>10mm) or if there are adverse weather conditions (snow storms, typhoons, monsoons) during the dates, raise warnings for heavy outdoor activities and recommend indoor alternatives.
2. Specific regional rules: E.g. Climbing Mount Fuji is closed outside of July, August, and early September due to snow. If Fuji climbing is scheduled outside this window, flag it as a critical error.

If there are conflicts, output them in this format:
WEATHER ERROR: [Description of conflict] | Recommendation: [Specific indoor alternative]

If there are no weather or seasonal conflicts, output "NO WEATHER ERRORS". Do not write any other conversational text.
"""
    response = llm.invoke(prompt)
    output = get_content(response)
    
    if "WEATHER ERROR" in output:
        validation_errors.append(output.strip())
        
    return {
        "validation_errors": validation_errors
    }


# Node 5: Budget Tracker Agent
def budget_tracker_node(state: AgentState) -> Dict[str, Any]:
    """
    Budget Tracker Persona: Deterministically parses exact cost annotations [Cost: ₹XXX]
    and calculates totals against the maximum budget limit.
    """
    if state.get("user_request") == "__OUT_OF_SCOPE__":
        return {
            "budget_errors": []
        }

    draft = state["draft_itinerary"]
    max_budget = state["max_budget"]
    budget_errors = []
    
    cost_matches = re.findall(r'\[Cost:\s*(?:[₹$]|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)\]', draft, re.IGNORECASE)
    
    total_cost = 0.0
    for match in cost_matches:
        cost_val = float(match.replace(",", ""))
        total_cost += cost_val
        
    if total_cost > max_budget:
        overage = total_cost - max_budget
        budget_errors.append(
            f"Budget Exception: Total planned cost of ₹{total_cost:.2f} exceeds the "
            f"maximum budget constraint of ₹{max_budget:.2f} by ₹{overage:.2f}. "
            f"Please adjust hotel selections, choose budget dining, or reduce paid activities."
        )
        
    return {
        "budget_errors": budget_errors
    }


# Transition/Routing State-Update Node
def loop_back_node(state: AgentState) -> Dict[str, Any]:
    """
    Transition logic node: Compiles existing error lists into a detailed feedback
    block, appends them to draft_itinerary, clears the active error lists, and increments iteration.
    """
    feedback = "\n\n--- FEEDBACK FROM VALIDATION LOOP (ITERATION {}) ---\n".format(state["iteration_count"] + 1)
    
    if state["validation_errors"]:
        feedback += "### Constraint & Geographic Validation Errors:\n"
        for err in state["validation_errors"]:
            feedback += f"- {err}\n"
            
    if state["budget_errors"]:
        feedback += "### Budget Discrepancies:\n"
        for err in state["budget_errors"]:
            feedback += f"- {err}\n"
            
    revised_draft = state["draft_itinerary"] + feedback
    
    return {
        "draft_itinerary": revised_draft,
        "validation_errors": [],  
        "budget_errors": [],      
        "iteration_count": state["iteration_count"] + 1
    }


# Node 6: Formatter & Exporter Agent
def formatter_exporter_node(state: AgentState) -> Dict[str, Any]:
    """
    Formatter Persona: Standardizes the verified itinerary draft. Exposes elegant Markdown and structured JSON.
    """
    draft = state["draft_itinerary"]
    if state.get("user_request") == "__OUT_OF_SCOPE__":
        return {
            "final_itinerary_markdown": draft,
            "final_itinerary_json": json.dumps({"error": "Out of scope request", "message": draft}, indent=2)
        }

    max_budget = state["max_budget"]
    month = state["travel_month"]
    destination = state["user_request"]
    
    llm = get_llm("structured")
    
    # 1. Generate beautiful Markdown Guide
    markdown_prompt = f"""
You are the Formatter & Exporter Agent. Create an elegant, reader-friendly Markdown travel guide for the verified itinerary.
Add emojis, clear subheaders, tables for expenses, and bold warning callouts for weather or permits.
Remove any raw feedback sections at the end of the itinerary.

CRITICAL: Keep all costs in Indian Rupees (₹). Do NOT convert them to Dollars ($) or use the Dollar symbol. Every price must use the ₹ symbol.

Verified Itinerary:
{draft}
"""
    markdown_guide = get_content(llm.invoke(markdown_prompt))
    
    # 2. Generate Structured JSON
    json_prompt = f"""
Convert the following travel itinerary into a structured JSON payload for database synchronization.
The JSON must strictly conform to this schema:
{{
  "destination": "String representing destination/vibe",
  "month": "String representing travel month",
  "max_budget": float,
  "total_estimated_cost": float,
  "days": [
    {{
      "day": int,
      "day_of_week": "String (e.g. Monday)",
      "theme": "Theme of the day",
      "activities": [
        {{
          "activity": "Activity name/details",
          "cost": float,
          "location": "Landmark/location name"
        }}
      ]
    }}
  ]
}}

Ensure that all cost numbers are floats. Output ONLY the JSON block. Do not include markdown code block characters like ```json.

Itinerary:
{draft}
"""
    json_output_str = get_content(llm.invoke(json_prompt)).strip()
    
    try:
        clean_json_str = json_output_str.replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(clean_json_str)
        final_json = json.dumps(parsed_json, indent=2)
    except Exception:
        final_json = json.dumps({
            "destination": destination,
            "month": month,
            "max_budget": max_budget,
            "error": "Failed to parse itinerary structure into standard JSON.",
            "raw_text": draft
        }, indent=2)
        
    cleaned_markdown = convert_usd_to_inr_in_text(markdown_guide)
    return {
        "final_itinerary_markdown": cleaned_markdown,
        "final_itinerary_json": final_json
    }


# -------------------------------------------------------------
# 4. Graph Compile & Routing Logic
# -------------------------------------------------------------
def route_after_budget(state: AgentState) -> str:
    has_validation_errors = len(state.get("validation_errors", [])) > 0
    has_budget_errors = len(state.get("budget_errors", [])) > 0
    
    if (has_validation_errors or has_budget_errors) and state["iteration_count"] < 3:
        return "loop_back"
    return "formatter_exporter"


def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("security_scrubber", security_scrubber_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("weather_adaptor", weather_adaptor_node)
    workflow.add_node("budget_tracker", budget_tracker_node)
    workflow.add_node("loop_back", loop_back_node)
    workflow.add_node("formatter_exporter", formatter_exporter_node)
    
    # Add Edges
    workflow.set_entry_point("security_scrubber")
    workflow.add_edge("security_scrubber", "planner")
    workflow.add_edge("planner", "validator")
    workflow.add_edge("validator", "weather_adaptor")
    workflow.add_edge("weather_adaptor", "budget_tracker")
    
    # Conditional routing after Budget Tracker Node
    workflow.add_conditional_edges(
        "budget_tracker",
        route_after_budget,
        {
            "loop_back": "loop_back",
            "formatter_exporter": "formatter_exporter"
        }
    )
    
    workflow.add_edge("loop_back", "planner")
    workflow.add_edge("formatter_exporter", END)
    
    return workflow.compile()

compiled_graph = build_agent_graph()
