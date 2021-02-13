import base64
import streamlit as st

@st.cache(allow_output_mutation=True)
def get_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def page_bg(png_file):
    bin_str = get_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-repeat: no-repeat;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    return