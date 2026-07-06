import os
import re
import sys
import json
import math
import requests
import datetime
from typing import Optional, List, Any, Tuple, Dict
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
from langchain_core.tools import tool

# Load environment
load_dotenv(override=True)

# Clean trailing spaces in environment
for k in list(os.environ.keys()):
    if k.strip() != k:
        os.environ[k.strip()] = os.environ[k]

hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HuggingFaceHub_API_Token")
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token.strip()
else:
    print("[warning] HUGGINGFACEHUB_API_TOKEN was not found in environment variables. Offline Mock/Demo mode is available.")


class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "mock_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt_lower = prompt.lower()
        
        # 1. Scope Check
        if "scope gatekeeper" in prompt_lower:
            # If prompt has out of scope trigger, return False
            if "sort an array" in prompt_lower or "how to bake" in prompt_lower or "write a python script" in prompt_lower:
                return '{"in_scope": false}'
            return '{"in_scope": true}'
            
        # 2. Coordinates Geocoder
        if "find the latitude and longitude" in prompt_lower:
            return '{"latitude": 19.8762, "longitude": 75.3433}'
            
        # 3. Main Destination Extractor
        if "identify the primary destination" in prompt_lower:
            return "Aurangabad"
            
        # 4. Metadata extractor
        if "analyze the user request and any existing itinerary draft. extract:" in prompt_lower:
            return '{"start_city": "Nagpur", "destination": "Aurangabad", "landmarks": ["Ajanta Caves", "Ellora Caves"]}'
            
        # 5. Landmark scheduling analysis
        if "parse the following travel itinerary and extract all landmark/location visits" in prompt_lower:
            if "monday" in prompt_lower:
                return """{
  "visits": [
    {
      "landmark": "Ajanta Caves",
      "day_number": 1,
      "day_of_week": "Monday"
    },
    {
      "landmark": "Ellora Caves",
      "day_number": 2,
      "day_of_week": "Tuesday"
    }
  ]
}"""
            else:
                return """{
  "visits": [
    {
      "landmark": "Ajanta Caves",
      "day_number": 1,
      "day_of_week": "Tuesday"
    },
    {
      "landmark": "Ellora Caves",
      "day_number": 2,
      "day_of_week": "Wednesday"
    }
  ]
}"""

        # 6. Weather Adaptor
        if "weather adaptor agent" in prompt_lower:
            return "NO WEATHER ERRORS"
            
        # 7. Planner Node - Revision Loop
        if "revise the previous travel itinerary" in prompt_lower:
            return """# Aurangabad Travel Plan (Revised & Corrected)

## Day 1 (Tuesday)
* Morning: Drive Nagpur to Aurangabad [Cost: ₹2000]
* Afternoon: Visit **Ajanta Caves** (Rescheduled to Tuesday because it is closed on Mondays) [Cost: ₹500]
* Dinner at hotel [Cost: ₹600]

## Day 2 (Wednesday)
* Visit **Ellora Caves** [Cost: ₹400]
* Shopping at local handicraft market [Cost: ₹500]

*Note: Total accommodation cost [Cost: ₹4500]*"""

        # 8. Planner Node - Fresh Draft
        if "lead travel planner agent" in prompt_lower:
            return """# Aurangabad Travel Plan Draft

## Day 1 (Monday)
* Morning: Drive Nagpur to Aurangabad [Cost: ₹2000]
* Afternoon: Visit **Ajanta Caves** (Note: this is closed on Mondays, but validator will fix it!) [Cost: ₹500]
* Dinner at hotel [Cost: ₹600]

## Day 2 (Tuesday)
* Visit **Ellora Caves** [Cost: ₹400]
* Shopping at local market [Cost: ₹500]

*Note: Total accommodation cost [Cost: ₹4500]*"""

        # 9. Formatter Node
        if "formatter & exporter agent" in prompt_lower:
            return """# ✈️ Verified Aurangabad Travel Itinerary

Welcome to your verified Aurangabad travel plan! 

### 🗓️ Daily Details

| Day | Activities | Cost | Status |
| :--- | :--- | :--- | :--- |
| **Day 1 (Tuesday)** | Drive Nagpur to Aurangabad | ₹2,000 | Planned |
| | Visit the heritage **Ajanta Caves** | ₹500 | **Verified Open** |
| | Dinner at hotel | ₹600 | Included |
| **Day 2 (Wednesday)** | Visit **Ellora Caves** | ₹400 | **Verified Open** |
| | Shopping at local handicraft market | ₹500 | Planned |

### 💰 Total Budget Breakdown
* **Accommodation**: ₹4,500
* **Activities & Transport**: ₹2,900
* **Meals**: ₹1,200
* **Total Estimated Cost**: ₹8,600

> ⚠️ **Verification Action**: The planner originally scheduled **Ajanta Caves** on Day 1 (Monday). The validator detected a temporal collision because Ajanta Caves is closed on Mondays, and automatically rescheduled it to Tuesday. All coordinates are verified to be within 40km, ensuring daily feasibility!"""

        # 10. JSON Exporter Node
        if "convert the following travel itinerary into a structured json payload" in prompt_lower:
            return """{
  "destination": "Aurangabad",
  "month": "July 2026",
  "max_budget": 10000.0,
  "total_estimated_cost": 8600.0,
  "days": [
    {
      "day": 1,
      "day_of_week": "Tuesday",
      "theme": "Heritage",
      "activities": [
        {"activity": "Drive Nagpur to Aurangabad", "cost": 2000.0, "location": "Aurangabad"},
        {"activity": "Ajanta Caves visit", "cost": 500.0, "location": "Ajanta Caves"}
      ]
    },
    {
      "day": 2,
      "day_of_week": "Wednesday",
      "theme": "History & Shopping",
      "activities": [
        {"activity": "Ellora Caves visit", "cost": 400.0, "location": "Ellora Caves"},
        {"activity": "Shopping", "cost": 500.0, "location": "Aurangabad"}
      ]
    }
  ]
}"""

        return "Mock response"


class HFInferenceClientLLM(LLM):
    model_name: str
    api_token: str

    @property
    def _llm_type(self) -> str:
        return "hf_inference_client"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        current_model = os.environ.get("HF_MODEL_NAME", self.model_name)
        if "mock" in current_model.lower():
            return MockLLM()._call(prompt, stop, run_manager, **kwargs)

        client = InferenceClient(api_key=self.api_token)
        messages = [{"role": "user", "content": prompt}]
        
        resp = client.chat.completions.create(
            model=current_model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048)
        )
        return resp.choices[0].message.content


def get_llm(model_type: str = "default"):
    """
    Exclusively initializes the Hugging Face InferenceClient LLM or the Mock LLM.
    Uses HF_MODEL_NAME environment variable if set by user selector,
    otherwise defaults to Qwen 2.5 72B.
    """
    load_dotenv(override=True)
    
    # Strip any potential spaces from variables loaded in os.environ
    for k in list(os.environ.keys()):
        if k.strip() != k:
            os.environ[k.strip()] = os.environ[k]
            
    model_name = os.environ.get("HF_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:fastest")
    if "mock" in model_name.lower():
        print(f"[get_llm] Initializing Mock Offline Demo LLM Mode")
        return MockLLM()

    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HuggingFaceHub_API_Token")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN was not found in environment variables. Please check your .env file.")
        
    print(f"[get_llm] Initializing Custom HF InferenceClient LLM: {model_name} | Model Type: {model_type}")
    return HFInferenceClientLLM(
        model_name=model_name,
        api_token=hf_token.strip(),
        temperature=0.7 if model_type == "default" else 0.01
    )


def get_content(response) -> str:
    """
    Standardizes LLM outputs by extracting the text content whether the response
    is a raw string or a chat message object.
    """
    if hasattr(response, "content"):
        return response.content
    return str(response)


def resolve_coordinates_via_llm(location: str) -> Tuple[float, float] or None:
    """
    Geocodes a location name to lat/long coordinates using the active LLM.
    """
    try:
        llm = get_llm("structured")
        prompt = f"""
Find the latitude and longitude coordinates for the following location: "{location}".
Output format: Output ONLY a raw JSON object with keys "latitude" and "longitude" (both floats). 
Do not add any markdown, comments, or extra text. Example:
{{"latitude": 48.8584, "longitude": 2.2945}}
"""
        response = llm.invoke(prompt)
        text = get_content(response).strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return float(data["latitude"]), float(data["longitude"])
    except Exception:
        return None


def get_usd_to_inr_rate() -> float:
    """
    Fetches the real-time USD to INR exchange rate from a free keyless API.
    Falls back to 83.0 if the network request fails.
    """
    try:
        res = requests.get("https://open.er-api.com/v6/latest/USD", timeout=3)
        if res.status_code == 200:
            data = res.json()
            rate = data.get("rates", {}).get("INR")
            if rate:
                print(f"[currency] Fetched real-time USD to INR rate: {rate}")
                return float(rate)
    except Exception as e:
        print(f"[currency] Failed to fetch real-time USD-to-INR rate: {e}. Using fallback 83.0")
    return 83.0


def convert_usd_to_inr_in_text(text: str) -> str:
    """
    Finds all occurrences of dollar values ($Amount) and converts them to equivalent Indian Rupees (₹)
    using the real-time USD to INR exchange rate, rounded to look realistic, and cleans up residual '$' signs.
    """
    rate = get_usd_to_inr_rate()
    def repl(match):
        val = float(match.group(1).replace(",", ""))
        inr_val = int(val * rate)
        if inr_val > 1000:
            inr_val = round(inr_val, -2)
        else:
            inr_val = round(inr_val, -1)
        return f"₹{inr_val:,}"
        
    # Match $150 or $150.00
    text = re.sub(r'\$\s*([\d,]+(?:\.\d+)?)', repl, text)
    text = text.replace("$", "₹")
    return text


def search_ddg(query: str) -> str:
    """
    Performs a DuckDuckGo text search and returns the top 3 results formatted.
    """
    print(f"[search_ddg] Querying DuckDuckGo: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                summary = []
                for r in results:
                    summary.append(f"- {r.get('title')}: {r.get('body')} (Source: {r.get('href')})")
                return "\n".join(summary)
    except Exception as e:
        print(f"[search_ddg] Error during search: {e}")
        return f"Search failed: {str(e)}"
    return "No search results found."


def enrich_context_with_web_searches(user_request: str, draft_itinerary: str = "") -> str:
    """
    Extracts origin, destination, and landmark lists using a quick structured LLM check,
    queries DuckDuckGo for fares, hotels, and landmark constraints, and returns a formatted context block.
    """
    llm = get_llm("structured")
    
    extract_prompt = f"""
Analyze the user request and any existing itinerary draft. Extract:
- start_city: The origin city of the trip (e.g. "Nagpur").
- destination: The main destination city or region of the trip (e.g. "Aurangabad").
- landmarks: A list of specific landmarks or attractions to visit (e.g. ["Ajanta Caves", "Ellora Caves"]).

User Request: {user_request}
Itinerary Draft (if any): {draft_itinerary}

Output format: Output ONLY a valid JSON object with keys "start_city" (string), "destination" (string), and "landmarks" (list of strings).
Do not add markdown code blocks (no ```json) or any other text.
Example:
{{"start_city": "Mumbai", "destination": "Pune", "landmarks": ["Shaniwar Wada", "Aga Khan Palace"]}}
"""
    try:
        response = llm.invoke(extract_prompt)
        text = get_content(response).strip().replace("```json", "").replace("```", "").strip()
        meta = json.loads(text)
    except Exception as e:
        print(f"[enrich_context] Metadata extraction failed: {e}")
        meta = {"start_city": "", "destination": "", "landmarks": []}
        
    start_city = meta.get("start_city", "").strip()
    destination = meta.get("destination", "").strip()
    landmarks = meta.get("landmarks", [])
    
    search_context = []
    
    # Search 1: Transport Fares
    if start_city and destination:
        q = f"{start_city} to {destination} travel fare options cost train bus ticket in Rupees"
        fares = search_ddg(q)
        search_context.append(f"### Real-time Transportation Fares ({start_city} to {destination}):\n{fares}")
        
    # Search 2: Hotel & Food Costs
    if destination:
        q = f"{destination} average budget hotel price per night restaurant meal cost in Rupees"
        hotels = search_ddg(q)
        search_context.append(f"### Real-time Lodging & Dining Estimates ({destination}):\n{hotels}")
        
    # Search 3: Landmark schedule constraints & ticket price
    for lmark in landmarks:
        if lmark.strip():
            q = f"{lmark.strip()} opening times ticket price closed days weekly holiday"
            closure = search_ddg(q)
            search_context.append(f"### Real-time Landmark Info ({lmark}):\n{closure}")
            
    if not search_context:
        return "No real-time search context was retrieved."
        
    return "\n\n".join(search_context)


@tool
def calculate_distance(loc1: str, loc2: str) -> str:
    """
    Calculates the geodesic distance between two locations in Japan using the Haversine formula.
    Use this tool to catch geographic inconsistencies (e.g. visiting Tokyo and Kyoto on the same afternoon).
    
    Args:
        loc1: Name of the first location/city (e.g. 'Tokyo National Museum').
        loc2: Name of the second location/city (e.g. 'Kyoto Imperial Palace').
    """
    # Coordinates of common travel spots for distance validation
    COORDINATES: Dict[str, Tuple[float, float]] = {
        "tokyo": (35.6762, 139.6503),
        "kyoto": (35.0116, 135.7681),
        "osaka": (34.6937, 135.5023),
        "hakone": (35.2324, 139.1069),
        "nara": (34.6851, 135.8048),
        "hiroshima": (34.3853, 132.4553),
        "sapporo": (43.0618, 141.3545),
        "tokyo national museum": (35.7189, 139.7765),
        "kyoto imperial palace": (35.0254, 135.7621),
        "ghibli museum": (35.6962, 139.5704),
        "mount fuji": (35.3606, 138.7273),
        "sensoji temple": (35.7148, 139.7967),
        "kinkaku-ji": (35.0394, 135.7292),
        "shinjuku gyoen": (35.6852, 139.7100),
        "tokyo skytree": (35.7101, 139.8107)
    }
    
    def resolve_coords(name: str) -> Tuple[float, float] or None:
        nl = name.lower().strip()
        sorted_keys = sorted(COORDINATES.keys(), key=len, reverse=True)
        for k in sorted_keys:
            if k in nl or nl in k:
                return COORDINATES[k]
        name_words = set(nl.split())
        for k in sorted_keys:
            k_words = set(k.split())
            if len(k_words.intersection(name_words)) >= 2:
                return COORDINATES[k]
        return None
        
    c1 = resolve_coords(loc1) or resolve_coordinates_via_llm(loc1)
    c2 = resolve_coords(loc2) or resolve_coordinates_via_llm(loc2)
    
    if not c1 or not c2:
        return f"Warning: Could not determine coordinates for calculation between '{loc1}' and '{loc2}'."
        
    lat1, lon1 = c1
    lat2, lon2 = c2
    
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    status = "Feasible"
    if distance > 150:
        status = "NOT FEASIBLE (Too far to visit on the same day without dedicating the entire day to travel)"
    elif distance > 50:
        status = "Requires bullet train (Shinkansen) or highway transit"
        
    return f"Geodesic Distance: {distance:.2f} km. Travel Viability: {status}."
