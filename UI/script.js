function initMap() {
    const map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: 28.6139, lng: 77.2090 }, // Default center (Delhi, India)
        zoom: 5
    });

    map.addListener("click", async (event) => {
        const lat = event.latLng.lat();
        const lon = event.latLng.lng();
        const depth = prompt("Enter earthquake depth (km):");

        if (!depth || isNaN(depth)) {
            alert("Invalid depth! Please enter a number.");
            return;
        }

        try {
            const response = await fetch(`/predict?latitude=${lat}&longitude=${lon}&depth=${depth}`);
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Update UI with results
            document.getElementById("prediction-result").innerHTML = `
                <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa; box-shadow: 0px 0px 10px rgba(0,0,0,0.2);">
                    <b>Latitude:</b> ${data.latitude} <br>
                    <b>Longitude:</b> ${data.longitude} <br>
                    <b>Depth:</b> ${data.depth} km <br>
                    <b style="color: red;">Earthquake Probability:</b> <strong>${data.probability}%</strong>
                </div>`;
        } catch (error) {
            console.error("Error fetching prediction:", error);
            alert("Error fetching earthquake prediction. Please try again.");
        }
    });
}
