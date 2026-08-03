# Context-Aware Message Router

An AI-powered WhatsApp notification routing system that intelligently classifies incoming messages into **Notify**, **Digest**, or **Mute** based on message content, conversation history, user context, and multimedia understanding.

Built for the **HackerRank Orchestrate Hackathon**, this project combines contextual reasoning, multimodal AI, and workflow orchestration to reduce notification overload while ensuring important messages are never missed.

---

##  Features

-  Context-aware notification routing
-  Processes text, image, and voice messages
-  Voice transcription using Groq Whisper
-  Image understanding using Vision LLMs
-  Uses conversation history for better decisions
-  User and business relationship awareness
-  Scam and spam detection
-  Forwarded message identification
-  Parallel message processing
-  LangGraph workflow orchestration

---

## Problem Statement

Modern messaging applications generate hundreds of notifications every day.

Most notifications are irrelevant while important ones often get buried.

This project builds an intelligent routing system that decides whether each incoming message should be:

- 🔔 **Notify** — Immediate attention required
- 📰 **Digest** — Include in daily summary
- 🔕 **Mute** — Ignore notification

The decision is made using multiple sources of context instead of relying only on message text.

---

# Architecture

```
                Incoming Message
                       │
                       ▼
          Context Aggregator
       (History + User Metadata)
                       │
                       ▼
             Media Processor
      ┌──────────────┴──────────────┐
      │                             │
 Voice Transcription          Image Captioning
  (Groq Whisper)               (Vision Model)
      │                             │
      └──────────────┬──────────────┘
                     ▼
            LangGraph Workflow
                     │
                     ▼
             Routing Agent
                     │
                     ▼
      Notify / Digest / Mute
```

---

# Dataset Used

The router combines information from multiple CSV files.

- messages.csv
- message_history.csv
- message_events.csv
- users.csv
- groups.csv
- group_members.csv
- business_accounts.csv
- user_business_history.csv
- daily_notification_summary.csv
- images.csv
- voice_notes.csv

Media files include

- Images
- Voice notes

---

# AI Pipeline

### 1. Context Aggregation

The system loads all contextual information once and creates constant-time lookup tables.

It retrieves

- Previous conversations
- User relationships
- Group information
- Business interactions
- Notification history
- Message events

---

### 2. Multimedia Processing

For messages containing media,

#### Voice Notes

- Groq Whisper Large V3
- Automatic speech transcription

#### Images

- Vision Language Model
- Image description generation
- Visual context extraction

---

### 3. Message Classification

The routing agent identifies message categories such as

- Personal
- Urgent
- Event
- Payment
- Business Update
- Promotion
- Greeting
- Forwarded
- Scam
- Spam

It also detects prompt injection attempts and phishing patterns.

---

### 4. Routing Decision

Each message receives

- Action
- Message type
- Confidence score
- Explanation
- Supporting evidence messages

Example

| Action | Description |
|---------|-------------|
| Notify | Important message requiring immediate attention |
| Digest | Useful but not urgent |
| Mute | Low priority or spam |

---

# Technologies Used

### Programming

- Python

### AI / LLM

- Groq API
- Whisper Large V3
- Llama Vision Model

### Frameworks

- LangGraph
- Python Dotenv

### Data Processing

- Pandas

### Concurrency

- ThreadPoolExecutor

---

# Project Structure

```
Context-Aware-Message-Router/

│
├── dataset/
│   ├── media/
│   │   ├── audio/
│   │   └── images/
│   ├── messages.csv
│   ├── message_history.csv
│   ├── users.csv
│   └── ...
│
├── data_loader.py
├── media_processor.py
├── router_agent.py
├── graph_router.py
├── main.py
├── test_runner.py
├── requirements.txt
├── output.csv
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Context-Aware-Message-Router.git

cd Context-Aware-Message-Router
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GROQ_API_KEY=YOUR_API_KEY

USE_GROQ=1

GROQ_VISION_MODEL=llama-3.2-11b-vision-preview
```

---

# Run

```bash
python main.py
```

or

```bash
python main.py --dataset dataset --output output.csv
```

---

# Output

The system generates an `output.csv` file.

Example

| message_id | action | message_type | confidence |
|------------|---------|--------------|------------|
| msg001 | notify | urgent | 0.97 |
| msg002 | digest | promotion | 0.82 |
| msg003 | mute | scam | 0.99 |

---

# Key Highlights

- Context-aware reasoning instead of keyword matching
- Multimodal AI for voice and images
- LangGraph workflow orchestration
- Parallel processing for scalability
- Explainable routing decisions
- Scam and phishing detection
- Evidence-backed predictions

---

# Future Improvements

- User personalization through reinforcement learning
- Learning notification preferences over time
- Multi-language support
- Real-time WhatsApp integration
- Vector memory for long-term context
- Fine-tuned notification ranking model

---

# License

This project was developed for educational and hackathon purposes.
