import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
st.set_page_config(page_title="MsbA-Pred", layout="wide")

def main_page():
    c1, c2 = st.columns([8, 1])
    with c1:
        # Add logo image at the top (centered)
        logo_image = Image.open('LOGO.png')
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{img}" style="max-width: 100%; height: 170px;">
            </div>
            """.format(img=base64.b64encode(open('LOGO.png', "rb").read()).decode()), unsafe_allow_html=True
        )

        #  marquee after the logo
        st.markdown(
            """
            <div style="width: 100%; background-color: yellow; padding: 0.2px;">
                <marquee behavior="scroll" direction="left" style="font-size:18px; color:blue;">
                    Welcome to MsbA-Pred! This application allows you to predict the bioactivity towards inhibiting the "MsbA - ATP-dependent lipid A-core flippase" with respect to quinoline derivatives, for the treatment aganist Gram-negative bacteria.  
                </marquee>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Create two columns with specified widths
        col1, col2 = st.columns([1, 4])  # Column 1 is narrower than Column 2

        # col1: Sidebar for file upload
        with col1:
            st.header('Upload data')
            uploaded_file = st.file_uploader("Upload your input file as smile notation", type=['txt'])
            st.markdown("""[Example input file](https://raw.githubusercontent.com/MohanChemoinformaticsLab/MsbAPred/main/Sample_Smiles_File.txt)""")

            if st.button('Predict'):
                if uploaded_file is not None:
                    load_data = pd.read_table(uploaded_file, sep=' ', header=None, names=['Notation', 'Identity'])
                    load_data.reset_index(drop=True, inplace=True)
                    load_data.index = load_data.index.astype(int) + 1
                    load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

                    st.session_state['load_data'] = load_data
                    st.session_state['show_results'] = True
                else:
                    st.error("Error! Please upload a file.")

        # col2: MsBAPred image and details
        with col2:
            #  background images
            images = ['mole7.PNG', 'dock.PNG', 'msba.PNG']  # Ensure these files exist in the directory
            background_images = [base64.b64encode(open(image, "rb").read()).decode() for image in images]

            #  CSS animation for the background images
            css = f"""
            <style>
            @keyframes backgroundAnimation {{
                0% {{ background-image: url(data:image/png;base64,{background_images[0]}); }}
                33% {{ background-image: url(data:image/png;base64,{background_images[1]}); }}
                67% {{ background-image: url(data:image/png;base64,{background_images[2]}); }}
                100% {{ background-image: url(data:image/png;base64,{background_images[0]}); }}
            }}
            .background {{
                animation: backgroundAnimation 10s infinite;
                background-size: cover;
                background-repeat: no-repeat;
                background-position: center;
                height: 50vh;  /* Adjust the height as needed */
            }}
            </style>
            """
       
            st.markdown(css, unsafe_allow_html=True)

            # Full introduction text
            full_text = """Designing new and efficient drugs through machine learning-assisted quantitative structure-activity relationships (ML-QSAR) has emerged as a promising strategy for addressing multidrug-resistant gram-negative bacterial infections. We developed robust ML-QSAR models using Support Vector Machine (SVM) with molecular fingerprints to predict the activity of experimentally known quinoline-based MsbA inhibitors and to guide the design of more potent antibacterial compounds. The molecular fingerprint-based SVM model was rigorously validated using internal and external metrics, supported by applicability domain analyses. MsbA-Pred, a platform-independent tool, allows users to predict the MsbA inhibitory activity of compounds from any device."""

            # Truncate the text to show only the first 30 words initially
            truncated_text = " ".join(full_text.split()[:30]) + "..."

            # Initialize session state for tracking whether the full text is shown
            if 'show_full_text' not in st.session_state:
                st.session_state.show_full_text = False

            # Custom HTML/CSS for styling the "Read More" and "Read Less" buttons
            button_css = """
            <style>
                .read-more-btn, .read-less-btn {
                    color: blue;
                    background: none;
                    border: none;
                    cursor: pointer;
                    text-decoration: underline;
                    display: inline;
                    font-size: 16px;
                }
                .read-more-btn:hover, .read-less-btn:hover {
                    color: darkblue;
            </style>
            """

            # Inject CSS styling into the Streamlit app
            st.markdown(button_css, unsafe_allow_html=True)

            # Display text based on the state
            if st.session_state.show_full_text:
                # Show full text and a "Read Less" button
                st.write(full_text)
                if st.button("Read Less"):
                    st.session_state.show_full_text = False
            else:
                # Show truncated text and a "Read More" button
                st.write(truncated_text)
                if st.button("Read More"):
                    st.session_state.show_full_text = True

            # Set the background using CSS for col2
            st.markdown(
                """
                <div class="background">
                    <div style="background-color: rgba(255, 255, 255, 0.5); height: 20%; width: 20%; position: absolute; top: 0; left: 0; z-index: 1;"></div>
                    <div style="position: relative; z-index: 2; padding: 20px; color: black;">
                        <h2>Application Details</h2>
                        <p>Here are some details about the app and its functionality:</p>
                        <ul>
                            <li><strong><em>Input</em></strong> : Text file containing smile notation.</li>
                            <li><strong><em>Output</em></strong>: Predictions based on the input data.</li>
                            <li><strong><em>Usage</em></strong> : Double click on the button to predict after uploading your data.</li>
                            <li><strong><em>Developed by Preena S Parvathy, Ratul Bhowmik, Anupama Binoy and Dr. C. Gopi Mohan.</em></strong></li>
                        </ul>
                        <p><strong>Note:</strong> Ensure that your input file is formatted correctly to avoid errors during processing.</p>
                        <h2>Credits</h2>
                        <p>- <em>Author affiliations: Bioinformatics and Computational Biology Lab, Amrita School of Nanosciences and Molecular Medicine, Amrita Vishwa Vidyapeetham, Kochi</em></p>
                        <p>- Descriptor calculated using <a href="http://www.yapcwsoft.com/dd/padeldescriptor/">PaDEL-Descriptor</a> <a href="https://doi.org/10.1002/jcc.21707">[Read the Paper]</a>.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    with c2:
        # Define button width to take full column width
        st.markdown("""
            <style>
            .full-width-button {
                width: 100%;
                font-size: 16px;
                padding: 5px;
                text-align: center;
                background-color: white;
                color: #333333;
                border:1px solid #D3D3D3;;
                cursor: pointer;
                border-radius: 5px; /* Optional: Adds rounded corners to the button */
                transition: background-color 0.3s ease; /* Optional: Adds hover effect */

            }
            .full-width-button:hover {
                color: red;  /* Text color becomes red on hover */
                border: 1px solid red;  /* Border becomes red on hover */
            }
            </style>
        """, unsafe_allow_html=True)

        # Define the "About Us" button-like link that opens the URL in a new tab
        about_button_html = """
        <a href="https://www.amrita.edu/school/nanosciences/" target="_blank">
            <button class="full-width-button" title="Click to show institutional details">
                About us
            </button>
        </a>
        """
        st.markdown(about_button_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


        # Define button width for "Contact Us". Button for showing/hiding author details
        button_clicked = st.button("Contact us", key="author_button", use_container_width=True, help="Click to show author details")

        # Track whether the button has been clicked using session state
        if "button_state" not in st.session_state:
            st.session_state.button_state = False  # Default: details hidden

        # Single click: Show details, Double click: Hide details
        if button_clicked:
            # Toggle state on button click
            if st.session_state.button_state:
                st.session_state.button_state = False  # Collapse details on second click
            else:
                st.session_state.button_state = True  # Expand details on first click
            

        # Display author details if the button is clicked (expanded state)
        if st.session_state.button_state:
            # Displaying the author's image
            author_image_path = 'author.PNG'  # Path to the author's image
            author_image = Image.open(author_image_path)

            # Display image
            st.image(author_image, caption="Dr. Gopi Mohan C.", use_column_width=True)

            # Display author bio
            st.markdown(
                """
                <h3>Dr. Gopi Mohan C.</h3>
                <p><strong>Professor</strong>, Amrita School of Nanosciences and Molecular Medicine, Kochi</p>
                <p><strong>Qualification:</strong> Ph.D</p>
                <p><strong>Email:</strong> <a href="mailto:cgmohan@aims.amrita.edu">cgmohan@aims.amrita.edu</a></p>
                <p><strong>Research Interests:</strong> Computational Biology & Structural Bioinformatics, Nanoinformatics, Protein Crystallography, Structure-Based Drug Design</p>
                <p>For more details, visit the <a href="https://www.amrita.edu/faculty/cgopimohan/" target="_blank">faculty page</a>.</p>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
     # Check if molecule.smi exists before attempting to delete
    if os.path.exists('molecule.smi'):
        os.remove('molecule.smi')
    else:
        st.warning("The file 'molecule.smi' does not exist. Skipping deletion.")

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    # Reads in saved regression model
    load_model = pickle.load(open('msba_rf.pkl', 'rb'))
   
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
   
    st.header('**Prediction output**')

    # DataFrame for predictions
    prediction_output = pd.Series(prediction, name='pIC50(M)')
   
    # Get the molecule names without extra indices
    molecule_name = st.session_state.load_data['Identity'].reset_index(drop=True)  # Reset index
   
    # Calculate IC50 from pIC50
    IC50_output = 10 ** (-prediction_output) * 1e6  
   
    # DataFrame combining molecule names, pIC50, and IC50
    df = pd.DataFrame({
        'molecule_name': molecule_name,
        'pIC50(M)': prediction_output.reset_index(drop=True),  # Reset index for predictions
        'IC50 (μM)': IC50_output.reset_index(drop=True)  # Add IC50 column
    })
   
    # Display the DataFrame in a table format
    df.index = df.index + 1
    st.write(df)  # Streamlit automatically renders DataFrame as a table.

# Function to display the results page
def results_page():


    # Go Back button 
    if st.button('Go Back'):
        st.session_state['show_results'] = False  # Set to False to show main page
        return  # Exit the function to prevent further processing
    # Function to display the results page
    load_data = st.session_state.load_data

    st.header('**Input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Apply trained model to make prediction on query compounds
    #st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptors_output.csv')
    desc.reset_index(drop=True, inplace=True)
    desc.index += 1
    #st.write(desc)
    #st.write(desc.shape)

    # Read descriptor list used in previously built model
    #st.header('**Subset of descriptors from previously built models**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    desc_subset.reset_index(drop=True, inplace=True)
    desc_subset.index += 1
    #st.write(desc_subset)
    #st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)

# Main loop
if 'show_results' not in st.session_state:
    st.session_state['show_results'] = False

if st.session_state.show_results:
    results_page()  # Show the results page
else:
    main_page()  # Show the main page initially