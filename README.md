# NeuroSynth — Real-Time EEG-to-Text Brain-Computer Interface

A seq2seq Transformer model that decodes raw EEG motor-imagery signals 
into text commands in real time.

## Stack
- Python · PyTorch · Transformers · MNE · FastAPI · React · WebSockets

## Project Structure

```
neurosynth/
├── data/                   # Raw + processed EEG data
├── notebooks/              # Colab notebooks for training
├── src/
│   ├── data/               # Data loading & preprocessing
│   ├── models/             # Transformer + encoder architecture
│   ├── training/           # Training loop, metrics
│   └── serving/            # FastAPI + WebSocket server
├── frontend/               # React UI
├── docker-compose.yml
└── requirements.txt

```


## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```