document.addEventListener("DOMContentLoaded", function () {
    console.log("DOM fully loaded and parsed"); // Debugging

    // Fetch dataset list on page load
    fetch("http://127.0.0.1:5000/list-datasets")
        .then((response) => response.json())
        .then((data) => {
            const datasetDropdown = document.getElementById("dataset");
            if (data.datasets) {
                data.datasets.forEach((file) => {
                    const option = document.createElement("option");
                    option.value = file;
                    option.textContent = file;
                    datasetDropdown.appendChild(option);
                });
                console.log("Datasets loaded:", data.datasets); // Debugging
            } else {
                alert("Error fetching datasets: " + data.error);
            }
        })
        .catch((error) => {
            console.error("Error fetching datasets:", error);
        });

    // Event listener for "Run Model" button
    document.getElementById("runModel").addEventListener("click", function () {
        console.log("Run Model button clicked"); // Debugging
        const action = "1";
        const dataset = document.getElementById("dataset").value;

        if (!dataset) {
            alert("Please select a dataset.");
            return;
        }

        const resultsDiv = document.getElementById("results");
        resultsDiv.innerHTML = "<p>Running Main.py...</p>";

        fetch("http://127.0.0.1:5000/run-main", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action, dataset }),
        })
            .then((response) => response.json())
            .then((data) => {
                console.log("Server response:", data); // Debugging
                if (data.error) {
                    resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                } else {
                    resultsDiv.innerHTML = `<h3>Output:</h3><pre>${data.output}</pre>`;
                }
            })
            .catch((error) => {
                console.error("Fetch error:", error); // Debugging
                resultsDiv.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
            });
    });

    // Event listener for "Run Limited" button
    document.getElementById("runLimited").addEventListener("click", function () {
        console.log("Run Limited button clicked"); // Debugging
        const songName = document.getElementById("songName").value;

        if (!songName) {
            alert("Please enter a song name.");
            return;
        }

        const resultsDiv = document.getElementById("results");
        resultsDiv.innerHTML = "<p>Running LimitedDataRedesign.py...</p>";

        fetch("http://127.0.0.1:5000/run-limited", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ song_name: songName }),
        })
            .then((response) => response.json())
            .then((data) => {
                console.log("Server response:", data); // Debugging
                if (data.error) {
                    resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                } else {
                    resultsDiv.innerHTML = `<h3>Output:</h3><pre>${data.output}</pre>`;
                }
            })
            .catch((error) => {
                console.error("Fetch error:", error); // Debugging
                resultsDiv.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
            });
    });
});


document.getElementById("mainBtn").addEventListener("click", function () {
    console.log("Run Main.py button clicked"); // Debugging
    document.getElementById("home-page").classList.add("hidden");
    document.getElementById("main-page").classList.remove("hidden");
});

document.getElementById("limitedBtn").addEventListener("click", function () {
    console.log("Run LimitedDataRedesign.py button clicked"); // Debugging
    document.getElementById("home-page").classList.add("hidden");
    document.getElementById("limited-page").classList.remove("hidden");
});

document.getElementById("backToHomeMain").addEventListener("click", function () {
    document.getElementById("main-page").classList.add("hidden");
    document.getElementById("home-page").classList.remove("hidden");
});

document.getElementById("backToHomeLimited").addEventListener("click", function () {
    document.getElementById("limited-page").classList.add("hidden");
    document.getElementById("home-page").classList.remove("hidden");
});
