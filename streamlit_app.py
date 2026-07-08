import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(
    page_title="LungSheba | লাং সেবা",
    page_icon="🫁",
    layout="centered"
)

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("codewithdark/vit-chest-xray")
    model = AutoModelForImageClassification.from_pretrained("codewithdark/vit-chest-xray")
    model.eval()
    return processor, model

processor, model = load_model()

LABEL_COLUMNS = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

CONDITION_INFO = {
    "Cardiomegaly": {
        "urgency": "🔴 HIGH",
        "bangla": "কার্ডিওমেগালি (হৃদপিণ্ড বড় হওয়া)",
        "action": "⚠️ See a cardiologist or physician URGENTLY this week.",
        "explanation": "কার্ডিওমেগালি মানে হৃদপিণ্ড স্বাভাবিকের চেয়ে বড় হয়ে গেছে। এটি হৃদরোগের একটি লক্ষণ হতে পারে। দ্রুত একজন হৃদরোগ বিশেষজ্ঞের সাথে পরামর্শ করুন। মনে রাখবেন, এটি শুধুমাত্র একটি প্রাথমিক ধারণা। সঠিক রোগ নির্ণয়ের জন্য একজন ডাক্তারের সাথে পরামর্শ করুন।"
    },
    "Edema": {
        "urgency": "🔴 HIGH",
        "bangla": "ইডিমা (ফুসফুসে পানি)",
        "action": "⚠️ Seek medical attention URGENTLY — possible fluid in lungs.",
        "explanation": "ইডিমা মানে ফুসফুসে বা শরীরের টিস্যুতে অতিরিক্ত তরল জমে গেছে। এটি শ্বাসকষ্টের কারণ হতে পারে। এটি একটি গুরুতর সমস্যা এবং দ্রুত চিকিৎসা প্রয়োজন। মনে রাখবেন, এটি শুধুমাত্র একটি প্রাথমিক ধারণা। সঠিক রোগ নির্ণয়ের জন্য একজন ডাক্তারের সাথে পরামর্শ করুন।"
    },
    "Consolidation": {
        "urgency": "🟡 MEDIUM",
        "bangla": "কনসোলিডেশন (ফুসফুসে ঘনত্ব)",
        "action": "📅 See a doctor within 2-3 days for further evaluation.",
        "explanation": "কনসোলিডেশন মানে ফুসফুসের কিছু অংশ বায়ুর পরিবর্তে তরল বা কোষ দিয়ে ভরে গেছে। এটি নিউমোনিয়া বা অন্য সংক্রমণের লক্ষণ হতে পারে। দ্রুত একজন ডাক্তারের সাথে পরামর্শ করুন। মনে রাখবেন, এটি শুধুমাত্র একটি প্রাথমিক ধারণা। সঠিক রোগ নির্ণয়ের জন্য একজন ডাক্তারের সাথে পরামর্শ করুন।"
    },
    "Pneumonia": {
        "urgency": "🔴 HIGH",
        "bangla": "নিউমোনিয়া (ফুসফুসের সংক্রমণ)",
        "action": "⚠️ See a doctor TODAY — pneumonia requires immediate treatment.",
        "explanation": "নিউমোনিয়া হলো ফুসফুসের একটি সংক্রমণ যা ব্যাকটেরিয়া বা ভাইরাসের কারণে হয়। এতে জ্বর, কাশি এবং শ্বাসকষ্ট হয়। সময়মতো অ্যান্টিবায়োটিক চিকিৎসা নিলে সম্পূর্ণ সুস্থ হওয়া যায়। মনে রাখবেন, এটি শুধুমাত্র একটি প্রাথমিক ধারণা। সঠিক রোগ নির্ণয়ের জন্য একজন ডাক্তারের সাথে পরামর্শ করুন।"
    },
    "No Finding": {
        "urgency": "🟢 NORMAL",
        "bangla": "কোনো অস্বাভাবিকতা পাওয়া যায়নি",
        "action": "✅ No abnormality detected. Maintain regular health checkups.",
        "explanation": "এক্স-রেতে কোনো উল্লেখযোগ্য অস্বাভাবিকতা পাওয়া যায়নি। তবে নিয়মিত স্বাস্থ্য পরীক্ষা করা উচিত। যদি কোনো শারীরিক সমস্যা অনুভব করেন তাহলে ডাক্তারের পরামর্শ নিন। মনে রাখবেন, এটি শুধুমাত্র একটি প্রাথমিক ধারণা। সঠিক রোগ নির্ণয়ের জন্য একজন ডাক্তারের সাথে পরামর্শ করুন।"
    },
}

st.title("🫁 LungSheba | লাংশেবা")
st.subheader("AI-powered Chest X-ray Abnormality Detector")

st.warning("""
⚠️ **Disclaimer:** This tool is for awareness only.
It is NOT a medical diagnosis. Always consult a qualified doctor.
""")

st.divider()

uploaded_file = st.file_uploader(
    "📸 Upload a Chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    with st.spinner("🔍 Analyzing X-ray... please wait"):
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)[0]

        results = [
            {"label": LABEL_COLUMNS[i], "score": probs[i].item()}
            for i in range(len(LABEL_COLUMNS))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[0]
        condition_name = top["label"]
        confidence = top["score"] * 100
        info = CONDITION_INFO.get(condition_name, CONDITION_INFO["No Finding"])

    st.divider()
    st.subheader("📊 Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Condition", condition_name)
        st.metric("In Bangla", info["bangla"])
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
        st.metric("Urgency Level", info["urgency"])

    st.markdown("### 👉 What to do next:")
    st.markdown(f"**{info['action']}**")

    st.divider()
    st.subheader("📝 বাংলায় ব্যাখ্যা")
    st.info(info["explanation"])

    st.divider()
    st.subheader("📊 All Condition Scores")
    for r in results:
        st.progress(
            min(r["score"], 1.0),
            text=f"{r['label']}: {r['score']*100:.1f}%"
        )

    st.divider()
    st.error("""
🚨 **If you are concerned, please visit:**
- Your nearest **Upazila Health Complex**
- **Dhaka Medical College Hospital** — Chest Medicine Dept
- **National Institute of Diseases of the Chest and Hospital (NIDCH)**
    """)
