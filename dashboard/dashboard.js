const ctx = document.getElementById("trafficChart");

const THRESHOLD = 7000;

const trafficChart = new Chart(ctx, {

type: "line",

data: {
labels: [],
datasets: [{
label: "Packets per Interval",
data: [],
borderColor: "#38bdf8",
fill: false,
tension: 0.3
}]
},

options: {

responsive: true,

plugins:{
legend:{
labels:{color:"white"}
}
},

scales:{
x:{ticks:{color:"white"}},
y:{ticks:{color:"white"}}
}

}

});
function updateChart(packetValue){
    const now = new Date().toLocaleTimeString();

    trafficChart.data.labels.push(now);
    trafficChart.data.datasets[0].data.push(packetValue);

    if(trafficChart.data.labels.length > 15){
        trafficChart.data.labels.shift();
        trafficChart.data.datasets[0].data.shift();
    }

    trafficChart.update();
}

setInterval(()=>{

fetch("traffic.json?t=" + Date.now())
.then(res=>res.json())
.then(data=>{
    updateChart(data.packets)
})

fetch("alert.json")
.then(res=>res.json())
.then(data=>{

    const statusElement = document.getElementById("deviceStatus")
    const threatElement = document.getElementById("threatLevel")

    if(data.status === "QUARANTINED"){
        statusElement.innerText = "QUARANTINED"
        statusElement.className = "quarantine"

        threatElement.innerText = "HIGH"
        threatElement.className = "danger"
    }else{
        statusElement.innerText = "NORMAL"
        statusElement.className = "normal"

        threatElement.innerText = "LOW"
        threatElement.className = "safe"
    }

})

},2000)