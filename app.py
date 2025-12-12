from flask import Flask, request, render_template, redirect, url_for
import os
import uuid
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the keras model
model = tf.keras.models.load_model('model.keras')

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)
    return img_array


def get_class_details(class_name):
    content = {
        'NonDemented': {
            'overview': "This stage indicates no signs of cognitive decline. The individual functions normally, with no memory loss or other symptoms of dementia. Brain scans show no evidence of the pathological changes associated with Alzheimer's disease.",
            'symptoms': "No symptoms are present. The individual has normal memory, judgment, and reasoning abilities.",
            'causes': "N/A - This is a healthy state. General causes of Alzheimer's involve the buildup of amyloid plaques and tau tangles in the brain, but these are not present at this stage.",
            'diagnosis': "A 'NonDemented' diagnosis is confirmed through cognitive tests, neurological exams, and brain imaging (MRI, PET scans) that show no signs of dementia or Alzheimer's-related changes.",
            'treatments': "The focus is on prevention. This includes maintaining a healthy lifestyle with regular exercise, a balanced diet (like the Mediterranean diet), active social engagement, and mentally stimulating activities.",
            'clinical_trials': "Individuals in this group may be eligible for prevention trials, which study interventions that could delay or prevent the onset of dementia in at-risk populations."
        },
        'VeryMildDemented': {
            'overview': "This is the earliest stage of cognitive decline. The individual may experience minor memory lapses, such as forgetting familiar words or the location of everyday objects. These symptoms are often subtle and may not be apparent to family, friends, or medical professionals.",
            'symptoms': "• Forgetting familiar words or names.\n• Losing or misplacing everyday objects.\n• Trouble remembering information that was just read.\n• Increased difficulty in planning or organizing.",
            'causes': "At this stage, the initial pathological changes of Alzheimer's, such as the formation of amyloid plaques, may be starting in the brain, particularly in areas involved in memory and learning.",
            'diagnosis': "Diagnosis can be challenging. A detailed medical interview may reveal subtle symptoms. Advanced imaging techniques like PET scans might detect early amyloid buildup, but symptoms are often attributed to normal aging.",
            'treatments': "No specific medications are approved for this very mild stage. Treatment focuses on lifestyle interventions, cognitive enhancement strategies, and monitoring the progression of symptoms.",
            'clinical_trials': "Many clinical trials focus on this 'preclinical' or 'mild cognitive impairment' stage, testing drugs that aim to clear amyloid plaques or prevent their formation to halt the disease's progression."
        },
        'MildDemented': {
            'overview': "In this stage, cognitive decline becomes more noticeable to family and friends. Memory problems and cognitive difficulties interfere with daily activities. The individual may get lost in familiar places or have trouble managing finances.",
            'symptoms': "• Significant memory loss, especially of recent events.\n• Getting lost in familiar surroundings.\n• Difficulty with complex tasks like paying bills.\n• Personality and behavioral changes, such as becoming withdrawn or moody.",
            'causes': "The buildup of plaques and tangles is more widespread, causing damage to brain cells in regions responsible for memory, thought, and planning. This damage leads to the noticeable cognitive and functional decline.",
            'diagnosis': "Diagnosis is typically made through a combination of patient interviews, cognitive testing (e.g., Mini-Mental State Exam), and brain scans (MRI) that may show brain shrinkage (atrophy) in certain areas.",
            'treatments': "Medications such as cholinesterase inhibitors (e.g., donepezil) or memantine may be prescribed to manage symptoms. Occupational therapy and creating a safe, structured environment are also crucial.",
            'clinical_trials': "Clinical trials for this stage often test new medications to slow disease progression, as well as non-pharmacological interventions like cognitive training or specific diets to improve quality of life."
        },
        'ModerateDemented': {
            'overview': "This is a stage of significant cognitive decline where the individual requires more assistance with daily activities. Confusion, memory loss, and personality changes are more pronounced. The person may not recognize family members.",
            'symptoms': "• Major gaps in memory; may forget personal history.\n• Inability to recall their own address or phone number.\n• Confusion about the date, time, or location.\n• Need for help with choosing proper clothing and performing daily tasks.\n• Significant personality changes, including paranoia or agitation.",
            'causes': "Widespread damage to brain cells affects language, reasoning, sensory processing, and conscious thought. Brain atrophy is typically more significant and visible on MRI scans.",
            'diagnosis': "Diagnosis is confirmed by the severity of symptoms and the level of functional impairment. Cognitive tests will show significant decline, and the individual is clearly unable to live independently and safely.",
            'treatments': "Treatment focuses on managing symptoms and ensuring safety. Medications may continue to be used. Behavioral interventions are key to managing agitation or aggression. Caregiver support and education are essential.",
            'clinical_trials': "Trials for this stage may focus more on managing behavioral symptoms and improving quality of life for both the patient and the caregiver, rather than on reversing the underlying disease process."
        }
    }
    return content.get(class_name, {})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        return redirect(url_for('index'))

    # Generate secure unique filename
    ext = os.path.splitext(secure_filename(file.filename))[1]
    unique_filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(filepath)
        processed_img = preprocess_image(filepath)
        prediction = model.predict(processed_img)[0]
        class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        pred_index = np.argmax(prediction)
        predicted_class = class_names[pred_index]
        confidence = float(prediction[pred_index]) * 100

        details = get_class_details(predicted_class)

        return render_template('result.html', prediction=predicted_class, confidence=confidence, details=details)
    except Exception as e:
        # Log error in production
        print(f"Prediction error: {e}")
        return redirect(url_for('index'))
    finally:
        # Clean up uploaded file after prediction
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
