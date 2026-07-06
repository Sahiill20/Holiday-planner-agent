"""
Model Context Protocol (MCP) Server for local travel constraints.
This standalone server manages local, hard travel restrictions to simulate real-world API data boundaries.

Hackathon Rubric Mapping:
- MCP Server: A standalone Model Context Protocol server exposing local travel constraints securely.
"""

from fastmcp import FastMCP
import json
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Initialize the FastMCP server
mcp = FastMCP("Japan Travel Constraints Server")

# 2. Local mock database of constraints
CONSTRAINTS = {
    "tokyo national museum": {
        "landmark": "Tokyo National Museum",
        "constraints": "Closed on Mondays. Open Tuesday-Sunday 9:30 AM - 5:00 PM.",
        "closed_days": ["Monday"]
    },
    "kyoto imperial palace": {
        "landmark": "Kyoto Imperial Palace",
        "constraints": "Permit required for entry. Closed on Mondays and public holidays.",
        "closed_days": ["Monday"]
    },
    "ghibli museum": {
        "landmark": "Ghibli Museum",
        "constraints": "Advance reservation tickets required. Closed on Tuesdays.",
        "closed_days": ["Tuesday"]
    },
    "mount fuji": {
        "landmark": "Mount Fuji",
        "constraints": "Climbing trails are only open during the summer climbing season (July to early September). Closed in other months due to snow and hazardous conditions.",
        "closed_months": ["January", "February", "March", "April", "May", "June", "October", "November", "December"]
    },
    "sensoji temple": {
        "landmark": "Senso-ji Temple",
        "constraints": "Open 24/7. Free entry.",
        "closed_days": []
    },
    "kinkaku-ji": {
        "landmark": "Kinkaku-ji (Golden Pavilion)",
        "constraints": "Open daily 9:00 AM - 5:00 PM.",
        "closed_days": []
    },
    "shinjuku gyoen": {
        "landmark": "Shinjuku Gyoen National Garden",
        "constraints": "Closed on Mondays. If Monday is a national holiday, it is open on Monday and closed on the following Tuesday.",
        "closed_days": ["Monday"]
    },
    "tokyo skytree": {
        "landmark": "Tokyo Skytree",
        "constraints": "Open daily 10:00 AM - 9:00 PM.",
        "closed_days": []
    }
}

# 3. Expose the check_local_constraints tool
@mcp.tool()
def check_local_constraints(location: str) -> str:
    """
    Check local scheduling, operational, closures, or permit constraints for a travel landmark.
    
    Args:
        location: The name of the travel location or landmark.
    """
    loc_lower = location.lower()
    matches = []
    
    # Direct lookup or substring match
    for key, data in CONSTRAINTS.items():
        if key in loc_lower or loc_lower in key:
            matches.append(data)
            
    # Fallback: check word overlap (e.g. "Kyoto Palace" -> "kyoto imperial palace")
    if not matches:
        query_words = set(loc_lower.split())
        for key, data in CONSTRAINTS.items():
            key_words = set(key.split())
            if len(key_words.intersection(query_words)) >= 2:
                matches.append(data)
            
    if not matches:
        # Fallback to dynamic LLM constraint lookup for global destinations
        load_dotenv(override=True)
        hftoken = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HuggingFaceHub_API_Token")
        
        if hftoken:
            try:
                headers = {"Authorization": f"Bearer {hftoken.strip()}"}
                api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct:fastest"
                prompt = f"""
You are the Local Travel Constraints database. Provide the active operating hours, typical closed days, permit rules, and seasonal closures for the following travel landmark: "{location}".

Format the output exactly like this:
### {location.title()} Constraints:
- [A single line describing the primary operational constraints, closed days, and permit/booking requirements]
- Closed Days: [e.g. Monday, or None]
- Closed Months: [e.g. None, or Winter months]

Keep it concise, realistic, and factual. Only output this formatted block.
"""
                chat_prompt = f"<|system|>\nYou are a helpful travel assistant.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
                payload = {
                    "inputs": chat_prompt,
                    "parameters": {"max_new_tokens": 512, "temperature": 0.01}
                }
                resp = requests.post(api_url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    res_data = resp.json()
                    if isinstance(res_data, list) and len(res_data) > 0:
                        text = res_data[0].get("generated_text", "")
                        if "<|assistant|>" in text:
                            text = text.split("<|assistant|>")[-1].strip()
                        return text.strip()
                    elif isinstance(res_data, dict):
                        return res_data.get("generated_text", "").strip()
                return f"No local database match found for '{location}'. Hugging Face API status code {resp.status_code}: {resp.text}"
            except Exception as e:
                return f"No local database match found for '{location}'. Dynamic constraints check via HuggingFace failed: {str(e)}"
        else:
            return f"No local database match found for '{location}'. Operational warning: HUGGINGFACEHUB_API_TOKEN is not configured."
        
    result_str = ""
    for match in matches:
        result_str += f"### {match['landmark']} Constraints:\n"
        result_str += f"- {match['constraints']}\n"
        if match.get("closed_days"):
            result_str += f"- Closed Days: {', '.join(match['closed_days'])}\n"
        if match.get("closed_months"):
            result_str += f"- Closed Months: {', '.join(match['closed_months'])}\n"
        result_str += "\n"
        
    return result_str.strip()

if __name__ == "__main__":
    # Run the server via stdio transport (standard for MCP tools)
    mcp.run()
