const loading_h2 = document.getElementById('loading');
const prediction_result = document.getElementById('prediction_result');
const prediction_btn = document.getElementById('prediction_btn');
const file_input = document.getElementById('image_upload');
const top5_image = document.getElementById('top5_image');

prediction_btn.addEventListener('click', async () => {

    if (!file_input.files.length) {
        prediction_result.style.color = '#F5C45E'
        prediction_result.innerText = "⚠️ Please select an image first.";
        return;
    }

    loading_h2.innerHTML = "⏳ Predicting...";
    loading_h2.style.color = '#F5C45E'
    prediction_result.innerText = "";
    console.log("loading text changed successfuly!");
    top5_image.style.display = "none";
    
    try {
        const formData = new FormData();
        formData.append('file', file_input.files[0]);
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: 'POST',
            body: formData
        });

        console.log("Response received:", response);

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Parsed response:", data);

        if (data.error) {
            prediction_result.innerText = `❌ Error: ${data.error}`;
        } else {
            top5_image.src = "";
            loading_h2.innerHTML = "";
            prediction_result.innerHTML = `🍽️ Prediction: <strong>${data.class}</strong> <br> 🎯 Confidence: <strong>${data.confidence}%</strong>`;
            prediction_result.style.color = '#BE3D2A'
            const timestamp = new Date().getTime();
            top5_image.src = `/static/images/top_5.png?t=${timestamp}`; // Set image path
            top5_image.style.display = "block"; // Make it visible
        }

    } catch (error) {
        console.error("Error in prediction:", error);
        prediction_result.innerText = "❌ Failed to get prediction.";
    }    
})