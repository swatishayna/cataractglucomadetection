import streamlit as st 
import base64
from pathlib import Path



def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

st.set_page_config(page_title="Home",
                   layout='wide',
                   page_icon='./images/home.png')

st.markdown("<p style='text-align: right; color: white;'> "+img_to_html('./images/kpmg.png')+"</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'> "+img_to_html('./images/national_emblem_resized.png')+"</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: blue;'>JHARKHAND HEALTH AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>KPMG DEMO</h3>", unsafe_allow_html=True)
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")

st.subheader("Cataract and Glaucoma Classification and Detection")
st.caption('This project demostrates Cataract and Glaucoma Classification and Detection through fundus Images')

# Content
st.markdown("""
### Classify and Detect Cataract and Glaucoma
- Detect and Classify Types of Cataract and Glaucoma from fundus image
- Detect Cataract and Glaucoma in Real-Time
 

Below give are the object that our model will detect and classify
1. Types of Cataract
2. Types of Glaucoma
           
            """)
