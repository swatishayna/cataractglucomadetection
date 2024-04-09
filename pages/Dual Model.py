import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from reportlab.lib.utils import ImageReader
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import io
from skimage import color, feature
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





def convert_to_grayscale(image_array):
    grayscale_image = Image.fromarray(image_array).convert('L')
    return np.array(grayscale_image)

def contrast_stretching(image_array, min_out=0, max_out=255, min_in=None, max_in=None):
    if min_in is None:
        min_in = image_array.min()
    if max_in is None:
        max_in = image_array.max()
    
    stretched = (image_array - min_in) * ((max_out - min_out) / (max_in - min_in)) + min_out
    return stretched.astype(np.uint8)

def convert_to_binary(image_array, threshold=128):
    binary_image = (image_array > threshold) * 255
    return binary_image.astype(np.uint8)

def calculate_white_spot_ratio(binary_image):
    total_pixels = binary_image.size
    white_pixels = np.count_nonzero(binary_image)
    ratio = white_pixels / total_pixels
    return ratio

def compute_saliency(image_array):
    if image_array.ndim == 2:  # Check if the image is grayscale
        # Expand the dimensions to create a 3-channel image
        image_array_rgb = np.expand_dims(image_array, axis=-1)
        image_array_rgb = np.repeat(image_array_rgb, 3, axis=-1)
    else:
        image_array_rgb = image_array

    # Convert the image to LAB color space
    lab_image = color.rgb2lab(image_array_rgb)
    
    # Extract the L channel (luminance)
    luminance = lab_image[:, :, 0]
    
    # Compute the Canny edges as the saliency map
    saliency_map = feature.canny(luminance)
    
    # Normalize the saliency map to the range [0, 255]
    saliency_map = (saliency_map * 255).astype(np.uint8)
    
    return saliency_map

def pseudocolor_saliency(saliency_map, cmap=plt.get_cmap('hot')):
    # Normalize the saliency map to the range [0, 1]
    normalized_saliency = saliency_map.astype(np.float32) / 255.0

    # Apply pseudocoloring
    pseudocolored_saliency = (cmap(normalized_saliency) * 255).astype(np.uint8)

    return pseudocolored_saliency, saliency_map




st.set_page_config(page_title="Cataract and Glaucoma Detection",
                   layout='wide',
                   page_icon='./images/object.png')
st.markdown("<p style='text-align: right; color: white;'> "+img_to_html('./images/kpmg.png')+"</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'> "+img_to_html('./images/national_emblem_resized.png')+"</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: blue;'>JHARKHAND HEALTH AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>KPMG DEMO</h3>", unsafe_allow_html=True)
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")

st.subheader('Cataract and Glaucoma Classification and Detection')
st.write('Please upload your fundus image to classify and detect')

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')

def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {"filename":image_file.name,
                        "filetype":image_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        # validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            st.success('VALID IMAGE file type (png or jpeg)')
            return {"file":image_file,
                    "details":file_details}
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png, jpg, jpeg')
            return None

# Function to create the PDF report with all images and object detection text
def create_pdf_report(image_array, pred_img, grayscale_img, contrast_stretched_img, binary_img, binary_img_eroded, binary_img_dilated, pseudocolored_img, pseudocolored_img2,white_spot_ratio, equivalent_diameter, pseudocolored_saliency, mean_magnitude,custom_image_path="pages/Labels_Guide.jpg"):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)

    labels_info1 = "Detection and Classification Report"
    c.setFont("Helvetica", 30)
    c.drawString(75, 750, labels_info1)
    
    # Calculate the center position for the image
    page_width, page_height = letter
    image_width = 300  # Adjust this based on your image size
    image_height = 300  # Adjust this based on your image size
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height + 300) / 2
    
    # Convert and save the predicted image as bytes
    img_bytes = BytesIO()
    pred_img_obj = Image.fromarray(pred_img)
    pred_img_obj.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Draw the image centered on the PDF
    c.drawImage(ImageReader(img_bytes), x_centered, y_centered, width=image_width, height=image_height)

    # Load and draw the custom image (if provided)
    if custom_image_path:
        custom_img = Image.open(custom_image_path)
        custom_img_width = 500  # Adjust this based on your image size
        custom_img_height = 300  # Adjust this based on your image size
        x_custom = (page_width - custom_img_width) / 2
        y_custom = 50  # Adjust this to position the custom image vertically
        # c.drawImage(ImageReader(custom_img), x_custom, y_custom, width=custom_img_width, height=custom_img_height)
    
    # Start a new page for the grayscale predicted image
    c.showPage()

    labels_info2 = "Grayscaled"
    c.setFont("Helvetica", 30)
    c.drawString(150, 600, labels_info2)

    # Calculate the center position for the grayscale image on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2

    # Add the grayscale predicted image to the new page
    grayscale_img_bytes = BytesIO()
    Image.fromarray(grayscale_img).save(grayscale_img_bytes, format='PNG')
    grayscale_img_bytes.seek(0)
    c.drawImage(ImageReader(grayscale_img_bytes), x_centered, y_centered, width=image_width, height=image_height)  # Adjust position as needed

    # Start another new page for the contrast-stretched grayscale image
    c.showPage()

    labels_info3 = "Contrast-Stretched"
    c.setFont("Helvetica", 30)
    c.drawString(150, 600, labels_info3)

    # Calculate the center position for the contrast-stretched image on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2
    

    # Add the contrast-stretched grayscale image to the new page
    contrast_stretched_img_bytes = BytesIO()
    Image.fromarray(contrast_stretched_img).save(contrast_stretched_img_bytes, format='PNG')
    contrast_stretched_img_bytes.seek(0)
    c.drawImage(ImageReader(contrast_stretched_img_bytes), x_centered, y_centered, width=image_width, height=image_height)  # Adjust position as needed

    # Start another new page for the binary predicted image
    c.showPage()

    labels_info4 = "Binary"
    c.setFont("Helvetica", 30)
    c.drawString(150, 600, labels_info4)

    # Calculate the center position for the binary image on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2

    # Add the binary predicted image to the new page
    binary_img_bytes = BytesIO()
    Image.fromarray(binary_img).save(binary_img_bytes, format='PNG')
    binary_img_bytes.seek(0)
    c.drawImage(ImageReader(binary_img_bytes), x_centered, y_centered, width=image_width, height=image_height)  # Adjust position as needed

    # Start another new page for the eroded binary predicted image
    c.showPage()

    labels_info5 = "Binary Eroded"
    c.setFont("Helvetica", 30)
    c.drawString(150, 600, labels_info5)

    # Calculate the center position for the eroded binary image on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2

    # Add the eroded binary predicted image to the new page
    binary_eroded_img_bytes = BytesIO()
    Image.fromarray(binary_img_eroded).save(binary_eroded_img_bytes, format='PNG')
    binary_eroded_img_bytes.seek(0)
    c.drawImage(ImageReader(binary_eroded_img_bytes), x_centered, y_centered, width=image_width, height=image_height)  # Adjust position as needed

    # Start another new page for the dilated binary predicted image
    c.showPage()

    labels_info5 = "Binary Dilated"
    c.setFont("Helvetica", 30)
    c.drawString(150, 600, labels_info5)

    # Calculate the center position for the dilated binary image on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2

    # Add the dilated binary predicted image to the new page
    binary_dilated_img_bytes = BytesIO()
    Image.fromarray(binary_img_dilated).save(binary_dilated_img_bytes, format='PNG')
    binary_dilated_img_bytes.seek(0)
    c.drawImage(ImageReader(binary_dilated_img_bytes), x_centered, y_centered, width=image_width, height=image_height)  # Adjust position as needed

    # Start another new page for the first pseudocolored image
    c.showPage()

    labels_info6 = "Pseudocolored"
    c.setFont("Helvetica", 30)
    c.drawString(150, 600, labels_info6)

    # Calculate the center position for the first pseudocolored image on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2

    # Add the first pseudocolored image to the new page
    pseudocolored_img_bytes = BytesIO()
    Image.fromarray(pseudocolored_img).save(pseudocolored_img_bytes, format='PNG')
    pseudocolored_img_bytes.seek(0)
    c.drawImage(ImageReader(pseudocolored_img_bytes), x_centered, y_centered, width=image_width, height=image_height)  # Adjust position as needed

    # Start another new page for the second pseudocolored image
    c.showPage()

    # Calculate the center position for the second pseudocolored image on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2

    # Add the second pseudocolored image to the new page
    pseudocolored_img2_bytes = BytesIO()
    Image.fromarray(pseudocolored_img2).save(pseudocolored_img2_bytes, format='PNG')
    pseudocolored_img2_bytes.seek(0)
    c.drawImage(ImageReader(pseudocolored_img2_bytes), x_centered, y_centered, width=image_width, height=image_height)  # Adjust position as needed


    c.showPage()

    labels_info7 = "Saliency Map"
    c.setFont("Helvetica", 30)
    c.drawString(160, 600, labels_info7)

    # Calculate the center position for the pseudocolored saliency map on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2

    # Add the pseudocolored saliency map to the new page
    pseudocolored_saliency_bytes = BytesIO()
    Image.fromarray(pseudocolored_saliency).save(pseudocolored_saliency_bytes, format='PNG')
    pseudocolored_saliency_bytes.seek(0)
    c.drawImage(ImageReader(pseudocolored_saliency_bytes), x_centered, y_centered, width=image_width, height=image_height)  # Adjust position as needed
    
    c.showPage()
    # ... (previous code)

    # Calculate the center position for the additional information on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2

    labels_info8 = "Diagnosis Results"
    c.setFont("Helvetica", 30)
    c.drawString(180, 750, labels_info8)

    # Display glaucoma severity and equivalent diameter
    c.setFont("Helvetica", 12)
    c.drawString(50, page_height - 120, f"Glaucoma Severity and Classification: {get_glaucoma_severity(equivalent_diameter)}")
    c.drawString(50, page_height - 140, f"Ratio: {equivalent_diameter:.2f}")

    c.setFont("Helvetica", 12)
    c.drawString(50, page_height - 160, f"Cataract Severity and Classification: {get_cataract_severity(mean_magnitude)}")
    c.drawString(50, page_height - 180, f"Mean Spectrum Magnitude: {mean_magnitude:.2f}")

    # Calculate the center position for the dilated binary image on the new page
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height) / 2

    # Draw the image centered on the PDF
    c.drawImage(ImageReader(img_bytes), 100,400, width=200, height=200)
    c.drawImage(ImageReader(pseudocolored_img_bytes), 300,200, width=200, height=200)
    c.drawImage(ImageReader(grayscale_img_bytes), 100,200, width=200, height=200)

    # Add the eroded binary predicted image to the new page
    binary_eroded_img_bytes = BytesIO()
    Image.fromarray(binary_img_eroded).save(binary_eroded_img_bytes, format='PNG')
    binary_eroded_img_bytes.seek(0)
    c.drawImage(ImageReader(binary_eroded_img_bytes), 300, 400, width=200, height=200)  # Adjust position as needed

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# Function to get glaucoma and  severity based on equivalent diameter
def get_glaucoma_severity(equivalent_diameter):
    if equivalent_diameter < 0.15:
        return "Non Glaucoma"
    elif 0.15 <= equivalent_diameter < 0.21:
        return "Primary Open Angle Glaucoma (Mild)"
    elif 0.21 <= equivalent_diameter < 0.30:
        return "Primary Open Angle Glaucoma (Moderate)"
    elif equivalent_diameter >= 0.30:
        return "Closed Angle Glaucoma (Severe)"
    
def get_cataract_severity(mean_magnitude):
    if mean_magnitude > 118:
        return ("Non Cataract")
    elif 115 <= mean_magnitude <= 118:
        return ("Hypermature Cataract")
    elif 105<= mean_magnitude < 115:
        return ("Mature Cataract")
    elif mean_magnitude < 104   :
        return ("Immature Cataract")

# def calculate_blurriness(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#     variance_of_laplacian = laplacian.var()
#     return variance_of_laplacian

def fourier_transform_processing(image_array):
    # Perform 2D Fourier Transform
    f_transform = np.fft.fft2(image_array)
    
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Compute the magnitude spectrum (log-scaled for better visualization)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    
    # Convert the dtype to uint8 for saving as an image
    magnitude_spectrum = (magnitude_spectrum / magnitude_spectrum.max() * 255).astype(np.uint8)
    
    return magnitude_spectrum



# Colormap for pseudocoloring
cmap = plt.get_cmap('jet')
cmap2 = plt.get_cmap('hsv')

def main():
    object = upload_image()
    
    if object:
        prediction = False
        image_obj = Image.open(object['file'])  
        image_array = np.array(image_obj)
        
        # # Calculate blurriness
        # blurriness = calculate_blurriness(image_array)     
        
        col1 , col2 = st.columns(2)
        
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)
            
        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Get Detection from YOLO')
            if button:
                with st.spinner("Getting Objects from image. Please wait..."):
                    # Below command will convert
                    # object to array
                    image_array = np.array(image_obj)
                    pred_img = yolo.predictions(image_array)
                    grayscale_pred_img = convert_to_grayscale(pred_img)
                    # Adjust min_out and max_out values for desired contrast
                    contrast_stretched_img = contrast_stretching(grayscale_pred_img, min_out=0, max_out=255)
                    binary_pred_img = convert_to_binary(contrast_stretched_img, threshold=230   )
                    
                    # Apply morphological operations (erosion and dilation)
                    kernel = np.ones((5, 5), np.uint8)  # Define a kernel
                    binary_pred_img_eroded = cv2.erode(binary_pred_img, kernel, iterations=1)
                    binary_pred_img_dilated = cv2.dilate(binary_pred_img_eroded, kernel, iterations=1)
                    # Calculate the white spot ratio
                    white_spot_ratio = calculate_white_spot_ratio(binary_pred_img_eroded)

                     # After contrast stretching, convert the grayscale image to a pseudocolored image
                    pseudocolored_img = cmap(contrast_stretched_img)
                    pseudocolored_img = (pseudocolored_img[:, :, :3] * 255).astype(np.uint8)

                    pseudocolored_img2 = cmap2(contrast_stretched_img)
                    pseudocolored_img2 = (pseudocolored_img2[:, :, :3] * 255).astype(np.uint8)

                    equivalent_diameter = white_spot_ratio * 100  # Assuming a scale factor, adjust as needed

                    # Pseudocolor the saliency map directly in the pseudocolor_saliency function
                    pseudocolored_saliency, saliency_map = pseudocolor_saliency(grayscale_pred_img)

                    # Ensure the pseudocolored_saliency array has the correct data type
                    pseudocolored_saliency = pseudocolored_saliency.astype(np.uint8)

                    # Find the coordinates of the pixel with the maximum saliency value
                    max_saliency_coords = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)

                    # Create a binary mask with a threshold to highlight the region
                    threshold = 0.9  # You may need to adjust this threshold based on your data
                    mask = (saliency_map > threshold * saliency_map.max()).astype(np.uint8)

                    # Apply the mask to the pseudocolored saliency map
                    highlighted_saliency = pseudocolored_saliency * mask[:, :, np.newaxis]



                    prediction = True
                
        if prediction:
            
            # ... (Displaying previous images)st.subheader("Predicted Image (Original)")
            st.caption("Object detection from YOLO V5 model")
            st.image(Image.fromarray(pred_img))  # Display the original predicted image

            st.subheader("Grayscale Predicted Image")
            st.image(Image.fromarray(grayscale_pred_img))  # Display the grayscale version

            st.subheader("Contrast-Stretched Grayscale Image")
            st.image(Image.fromarray(contrast_stretched_img))  # Display the contrast-stretched grayscale version

            st.subheader("Pseudocolored Grayscale Image")
            st.image(Image.fromarray(pseudocolored_img))
            st.image(Image.fromarray(pseudocolored_img2))

            # Display the pseudocolored Saliency Map
            st.subheader("Pseudocolored Saliency Map")
            st.image(Image.fromarray(pseudocolored_saliency))

            # Display the highlighted pseudocolored Saliency Map
            st.subheader("Highlighted Pseudocolored Saliency Map")
            st.image(Image.fromarray(highlighted_saliency))

            st.subheader("Binary Predicted Image")
            st.image(Image.fromarray(binary_pred_img))  # Display the binary version

            st.subheader("Binary Predicted Image (Dilation)")
            st.image(Image.fromarray(binary_pred_img_dilated))  # Display the dilated binary version

            st.subheader("Binary Predicted Image (Erosion)")
            st.image(Image.fromarray(binary_pred_img_eroded))  # Display the eroded binary version

            # st.text(f"Blurriness: {blurriness:.2f}")

            # # Check if the image is blurry based on a threshold (you may adjust the threshold)
            # blur_threshold = 350  # Adjust this threshold based on your preference
            # if blurriness < blur_threshold:
            #     st.success("Image is not blurry.")
            # else:
            #     st.warning("Image is blurry.")

            # st.subheader("Blurriness Detection Result:")
            # st.text(f"Blurriness: {blurriness:.2f}")


            st.subheader("Fourier Transform Result")
            magnitude_spectrum = fourier_transform_processing(contrast_stretched_img)
            st.image(Image.fromarray(magnitude_spectrum), caption="Magnitude Spectrum")

            st.subheader("Classification Results:")

             # Calculate and display the mean value of the magnitude spectrum
            mean_magnitude = np.mean(np.abs(magnitude_spectrum))
            st.text(f"Spectrum Value: {mean_magnitude:.2f}")

                # Check conditions for glaucoma severity
            if mean_magnitude > 118:
               st.success("Non Cataract")
            elif 115 <= mean_magnitude <= 118:
               st.error("Hypermature Cataract")
            elif 105<= mean_magnitude < 115:
               st.warning("Mature Cataract")
            elif mean_magnitude < 104   :
               st.warning("Immature Cataract")


            # Check conditions for glaucoma severity
            if equivalent_diameter > 0:
                # Display the white spot ratio as a diameter
                equivalent_diameter = white_spot_ratio * 100  # Assuming a scale factor, adjust as needed
                st.text(f"Glaucoma Ratio: {equivalent_diameter:.2f}")

                 # Check conditions for glaucoma severity
                if equivalent_diameter < 0.15:
                    st.success("Non Glaucoma")
                elif 0.15 <= equivalent_diameter < 0.21:
                    st.warning("Primary Open Angle Glaucoma (Mild)")
                elif 0.21 <= equivalent_diameter < 0.30:
                    st.warning("Primary Open Angle Glaucoma (Moderate)")
                elif equivalent_diameter >= 0.30:
                    st.error("Closed Angle Glaucoma (Severe)")
            else:
                st.warning("Non-Glaucoma")
            
            # Call the modified create_pdf_report function
            pdf_data = create_pdf_report(image_array, pred_img, grayscale_pred_img, contrast_stretched_img, binary_pred_img_eroded, binary_pred_img, binary_pred_img_dilated, pseudocolored_img, pseudocolored_img2, white_spot_ratio, equivalent_diameter,pseudocolored_saliency,mean_magnitude,custom_image_path="pages/Labels_Guide.jpg")
            st.download_button(label="Generate and Download PDF Report", data=pdf_data, file_name="detection_report.pdf", mime='application/pdf')

if __name__ == "__main__":
    main()
