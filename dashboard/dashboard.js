const ctx = document.getElementById("trafficChart").getContext("2d");

let labels = [];
let trafficData = [];

const chart = new Chart(ctx, {
    type: "line",
    data: {
        labels: labels,
        datasets: [{
            label: "Packets per Interval",
            data: trafficData,
            borderColor: "#38bdf8",
            fill: false,
            tension: 0.3
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { labels: { color: "white" } }
        },
        scales: {
            x: { ticks: { color: "white" } },
            y: { ticks: { color: "white" } }
        }
    }
});

function updateData() {

    fetch("http://localhost:5000/traffic?nocache=" + Date.now())
        .then(res => res.json())
        .then(data => {

            const time = new Date().toLocaleTimeString();
            labels.push(time);
            trafficData.push(data.packets);

            if (labels.length > 10) {
                labels.shift();
                trafficData.shift();
            }

            chart.update();
        });

    fetch("http://localhost:5000/alert?nocache=" + Date.now())
        .then(res => res.json())
        .then(alert => {

            document.getElementById("deviceStatus").innerText = alert.status;
            document.getElementById("threatLevel").innerText = alert.threat;
            document.getElementById("rfProb").innerText = alert.rf_prob;
            document.getElementById("isoScore").innerText = alert.iso_score;
            document.getElementById("modelDecision").innerText = alert.status;

            if (alert.status === "QUARANTINED") {
                document.getElementById("deviceStatus").style.color = "red";
            } else if (alert.status === "SUSPICIOUS") {
                document.getElementById("deviceStatus").style.color = "orange";
            } else {
                document.getElementById("deviceStatus").style.color = "lightgreen";
            }
        });
}

setInterval(updateData, 5000);