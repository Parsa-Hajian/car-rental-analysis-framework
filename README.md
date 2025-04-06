# Car Rental Analysis Framework

This project analyzes customer reviews from car rental companies to find patterns, improve services, and boost satisfaction. It uses Centauro at Milan Malpensa Airport as a case study.

### ***Caution*** : The data is dynamic. Whenever you execute the python script, the code will dynamically fetch, clean, and analyze data.

## 🔍 What’s Inside

- Scrapes real reviews from Booking.com and Trustpilot
- Cleans and analyzes the review text
- Runs sentiment analysis
- Generates insightful visualizations

## 📁 Folder Structure

- `src/`: Python script for the pipeline
- `outputs/`: Charts generated by the script
- `requirements.txt`: Dependencies

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/car_rental_case_study.py
```

Or open the notebook in `notebooks/`.

## 📊 Charts

- Sentiment distribution of reviews
- Word cloud of common terms
- Top words in negative reviews

## 📘 License

MIT License – open to all.
