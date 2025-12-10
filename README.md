# GuitarFlow 🎸

**A hands-on MLOps and Data Engineering portfolio project**

---

## What is This?

GuitarFlow is a binary image classifier that distinguishes electric guitars from acoustic guitars. The actual classification task itself is simple - intentionally. The real goal is to practice and demonstrate the **full ML engineering lifecycle**, from data collection to production deployment.

I'm using this project to transition from general software development back into my core domain: **ML and Data Engineering**.

---

## Why This Project?

I code, and I play acoustic guitar. So it's interesting. Also:

- **Relearn and practice** data engineering and MLOps workflows
- **Build something tangible** that showcases these skills
- **Document the process** as I go, making this a living portfolio piece

The guitar classifier is just the vehicle—what matters is the engineering practices around it.

---

## Project Stages

_Note: This README updates as the project progresses_

### 1. Data Collection 🚧
- Fetch guitar images via free APIs (Pixabay, Pexels, etc.)
- Build a simple database to manage image metadata
- Implement basic quality checks

### 2. Model Development 📋
- Train a baseline binary classifier
- Track experiments and results
- Version the model

### 3. Deployment 📋
- Deploy model to a free service (Streamlit or similar)
- Build a simple inference API
- Make it accessible for demo

### 4. Monitoring & Testing 📋
- Log predictions and performance
- Implement basic drift detection
- Set up automated testing

### 5. CI/CD Pipeline 📋
- Automate testing on commit
- Automate deployment on merge
- Keep it simple and reproducible

---

## Tech Stack

**Current:**
- Python 3.14 (shiny new things! and native uuid7 support!)
- `requests` - API calls
- `SQLite` - Image metadata
- `loguru` - Logging
- `pytest` - Testing

**Planned:**
- ML framework (TBD)
- Deployment platform (Streamlit or similar)
- CI/CD (GitHub Actions)

Stack will be updated as decisions are made during development.

---

## Current Status

✅ Repository initialized  
🚧 Data collection classes and pipelines are in progress  
📋 Everything else planned but not started

---

## Project Goals

By the end of this, I'll have demonstrated:

1. Building data pipelines from scratch
2. Training and versioning ML models
3. Deploying models to production
4. Monitoring model performance
5. Automating the entire workflow

---

## License

MIT License

---

_This is a learning project and portfolio piece. Updates happen as I build._