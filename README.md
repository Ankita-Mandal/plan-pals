# PlanPals 

A multi-agent system that magically crafts the perfect weekend plans based on weather and events!

## Overview

PlanPals is an intelligent weekend planner that uses Google's A2A (Agent-to-Agent) SDK to coordinate multiple specialized agents:

1. **Weather Agent** - Checks weather predictions for the upcoming weekend
2. **Event Finder Agent** - Searches Eventbrite and local calendars for upcoming events
3. **Host Agent** - Coordinates between agents and suggests optimal weekend plans based on weather conditions and user preferences
4. *(Future)* **Messenger Agent:** Sends finalized plans to friends.

## Architecture

The system follows a multi-agent architecture where each agent runs as a separate service:
- Weather Agent (Port 10001)
- Event Finder Agent (Port 10002)
- Host Agent (Port 10000)

## Setup

### Prerequisites
- Python 3.10+
- Google API credentials (either API key or Vertex AI project details)

### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/plan-pals.git
cd plan-pals
```

2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
   - Copy `.env.example` to `.env` in each agent directory
   - Add your Google API credentials
   - Configure agent URLs

### Running the Agents

Start each agent in a separate terminal:

```bash
# Terminal 1: Start Weather Agent
cd weather_agent
python -m __main__

# Terminal 2: Start Event Finder Agent
cd event_finder_agent
python -m __main__

# Terminal 3: Start Host Agent
cd host_agent
python -m __main__
```

Access the Host Agent UI at: http://localhost:10000

