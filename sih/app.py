import streamlit as st
from code import preprocessing

st.title("Depression Detection")
st.header("Detect your depression with our AI model")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")
uploaded_file1=st.file_uploader("Choose a brain MRI image...", type=".nii.gz")
uploaded_file2=st.file_uploader("Choose a brain Bval...", type=".bval")
uploaded_file3=st.file_uploader("Choose a brain Bvec...", type=".bvec")
uploaded_file4=st.file_uploader("Choose a brain Label...", type=".nii.gz")
uploaded_file5=st.file_uploader("Choose a brain Mask...", type=".nii.gz")

if uploaded_file1 is not None and uploaded_file2 is not None and uploaded_file3 is not None and uploaded_file4 is not None and uploaded_file5 is not None:
    st.text("Processing...")
    result=preprocessing(uploaded_file1,uploaded_file2,uploaded_file3,uploaded_file4,uploaded_file5)
    st.success(result)









