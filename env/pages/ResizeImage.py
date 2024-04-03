import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io

# Function to resize and annotate image
def resize_and_annotate_image(image, size, annotations):
    img = Image.open(image)
    img_resized = img.resize(size, resample=Image.LANCZOS)  # Use high-quality resampling method
    
    # Initialize drawing context
    draw = ImageDraw.Draw(img_resized)
    
    # Annotate the image with text
    for annotation in annotations:
        position = annotation['position']
        text = annotation['text']
        font_size = annotation.get('font_size', 20)
        font = ImageFont.truetype("arial.ttf", font_size)
        draw.text(position, text, fill="red", font=font)
    
    return img_resized

# Main function
def main():
    st.title("Image Resizer  with Annotation")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image")

        # Get user input for size
        width = st.number_input("Enter width for resized image:", value=100)
        height = st.number_input("Enter height for resized image:", value=100)

        # Checkbox for adding annotations
        add_annotations = st.checkbox("Add Annotations")

        if add_annotations:
            # Get user input for annotations
            num_annotations = st.number_input("Enter the number of annotations:", value=0)
            annotations = []
            for i in range(num_annotations):
                text = st.text_input(f"Enter text for annotation {i+1}:")
                x = st.number_input(f"Enter X-coordinate for annotation {i+1}:", value=0)
                y = st.number_input(f"Enter Y-coordinate for annotation {i+1}:", value=0)
                font_size = st.number_input(f"Enter font size for annotation {i+1}:", value=20)
                annotations.append({'text': text, 'position': (x, y), 'font_size': font_size})
        else:
            annotations = []  # No annotations if checkbox is unchecked

        if st.button("Resize and Annotate Image"):
            resized_annotated_image = resize_and_annotate_image(uploaded_image, (int(width), int(height)), annotations)
            st.image(resized_annotated_image, caption=f"Resized and Annotated Image ({width}x{height})")

            # Download resized and annotated image
            buffered = io.BytesIO()
            resized_annotated_image.save(buffered, format="PNG", quality=95)  # Set high compression quality
            resized_annotated_image_bytes = buffered.getvalue()
            st.download_button("Download Resized and Annotated Image", resized_annotated_image_bytes, file_name="resized_annotated_image.png")

if __name__ == "__main__":
    main()
