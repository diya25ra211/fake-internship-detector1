from html.parser import HTMLParser
import joblib
import pandas as pd
from PIL import Image
import pytesseract
import requests
import streamlit as st


# Built-in HTML text extractor
class HTMLTextStripper(HTMLParser):

  def __init__(self):
    super().__init__()
    self.reset()
    self.convert_charrefs = True
    self.text = []

  def handle_data(self, d):
    self.text.append(d)

  def get_data(self):
    return " ".join(self.text)


def strip_html_tags(html_content):
  stripper = HTMLTextStripper()
  stripper.feed(html_content)
  return stripper.get_data()


# Load the trained CatBoost model
model = joblib.load("catboost_model.pkl")

st.title("🚨 Fake Internship Postings Detector")
st.write(
    "Upload a screenshot or paste a link of the internship posting below to"
    " check if it's legitimate or a potential scam."
)

# Initialize session state for extracted text
if "extracted_text" not in st.session_state:
  st.session_state.extracted_text = ""

# Choose input method
input_method = st.radio(
    "Choose input method:", ("Upload Screenshot", "Paste Job Link (URL)")
)

if input_method == "Upload Screenshot":
  uploaded_file = st.file_uploader(
      "Upload internship screenshot", type=["png", "jpg", "jpeg"]
  )
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    if st.button("Extract Text from Screenshot"):
      with st.spinner("Processing OCR..."):
        try:
          st.session_state.extracted_text = pytesseract.image_to_string(image)
          st.success("Text extracted successfully!")
        except Exception as e:
          st.error(
              f"OCR Error: {e}. Make sure Tesseract is installed on your"
              " system."
          )

elif input_method == "Paste Job Link (URL)":
  job_url = st.text_input("Enter Internship Posting URL:")
  if st.button("Fetch Webpage Content"):
    if job_url:
      with st.spinner("Fetching webpage..."):
        try:
          headers = {
              "User-Agent": (
                  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
              )
          }
          response = requests.get(job_url, headers=headers, timeout=10)
          response.raise_for_status()
          st.session_state.extracted_text = strip_html_tags(response.text)
          st.success("Webpage content fetched successfully!")
        except Exception as e:
          st.error(f"Error fetching URL: {e}")
    else:
      st.warning("Please enter a valid URL.")

# Show text area preview so user can verify/edit extracted text
if st.session_state.extracted_text:
  st.subheader("Extracted Content Preview")
  st.session_state.extracted_text = st.text_area(
      "You can review or edit the text below before analyzing:",
      st.session_state.extracted_text,
      height=150,
  )

  if st.button("🔍 Check Internship Status"):
    with st.spinner("Analyzing posting features..."):
      text_lower = st.session_state.extracted_text.lower()
      job_desc_length = len(st.session_state.extracted_text)

      # Heuristic keyword scorers
      urgency_keywords = ["urgent", "immediate hiring", "hurry", "limited seats"]
      urgency_score = min(
          100.0, sum(text_lower.count(kw) for kw in urgency_keywords) * 25.0
      )

      phishing_keywords = [
          "telegram",
          "whatsapp only",
          "registration fee",
          "security deposit",
          "training fee",
      ]
      phishing_language_score = min(
          100.0, sum(text_lower.count(kw) for kw in phishing_keywords) * 30.0
      )

      vague_keywords = ["work from home", "easy money", "no experience needed"]
      vague_description_score = min(
          100.0, sum(text_lower.count(kw) for kw in vague_keywords) * 20.0
      )

      trust_keywords = ["certified", "registered company", "official partner"]
      trust_signal_score = max(
          10.0,
          100.0 - (sum(text_lower.count(kw) for kw in trust_keywords) * 20.0),
      )

      # Build data dictionary matching model features
      data = {
          "company_age": 15.0,
          "linkedin_presence": 1.0,
          "website_available": 1.0,
          "domain_age_months": 120,
          "verification_status": 1.0,
          "stipend": 25000,
          "payment_required": (
              1.0
              if any(k in text_lower for k in ["fee", "deposit", "pay"])
              else 0.0
          ),
          "registration_fee": (
              1.0 if "registration fee" in text_lower else 0.0
          ),
          "job_description_length": job_desc_length,
          "grammatical_errors": 1.0,
          "vague_description_score": vague_description_score,
          "urgency_score": urgency_score,
          "keyword_spam_score": 15.0,
          "fake_certificate_offer": 0.0,
          "recruiter_experience_years": 4.0,
          "suspicious_email_domain": 0.0,
          "recruiter_email_type_Free": 0.0,
          "recruiter_response_time_hours": 12.0,
          "social_media_presence": 1.0,
          "emotional_manipulation_score": 20.0,
          "phishing_language_score": phishing_language_score,
          "trust_signal_score": trust_signal_score,
          "posting_year": 2026,
          "posting_month": 6,
          "posting_day": 1,
          "posting_dayofweek": 2,
          "phishing_risk_tier": 2,
          "urgency_risk_tier": 2,
          "vague_risk_tier": 2,
          "total_suspicious_flags": 1,
      }

      # Categorical encodings
      for value in ["Full-Time", "Internship", "Part-Time"]:
        data[f"employment_type_{value}"] = int(value == "Internship")

      for value in ["Onsite", "Remote"]:
        data[f"work_mode_{value}"] = int(value == "Remote")

      industries = [
          "Cybersecurity",
          "E-Commerce",
          "EdTech",
          "FinTech",
          "Gaming",
          "Healthcare",
          "Marketing",
          "Software",
      ]
      for value in industries:
        data[f"industry_{value}"] = int(value == "Software")

      locations = [
          "Berlin",
          "Dubai",
          "London",
          "New York",
          "San Francisco",
          "Singapore",
          "Sydney",
          "Toronto",
      ]
      for value in locations:
        data[f"location_{value}"] = int(value == "New York")

      company_sizes = ["Medium", "Small", "Startup"]
      for value in company_sizes:
        data[f"company_size_{value}"] = int(value == "Startup")

      input_data = pd.DataFrame([data])

      try:
        expected_features = model.feature_names_
        input_data = input_data[expected_features]

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        fake_index = classes.index(1) if 1 in classes else 0
        probability = probabilities[fake_index] * 100

        st.markdown("---")
        if prediction == 1:
          st.error("⚠️ Warning: This posting is likely FAKE!")
          st.write(f"Estimated scam probability: {probability:.2f}%")
        else:
          st.success("✅ This posting appears to be Legitimate.")
          st.write(f"Estimated real probability: {100 - probability:.2f}%")

      except Exception as e:
        st.error(f"Prediction error: {e}")