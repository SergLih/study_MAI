var express = require('express');
var app = express();
var bodyParser = require('body-parser');
const fs = require('fs');


var points = [];

var PORT = process.env.PORT || 3000;

app.use(express.static(__dirname));
app.use(bodyParser.json());

fs.readFile("./points.json", "utf8", (err, jsonString) => {
  if (err) {
    console.log("Error reading file from disk:", err);
    return;
  }
  try {
    points = JSON.parse(jsonString);
    console.log(points.length + ' records loaded from file'); // => "Customer address is: Infinity Loop Drive"
  } catch (err) {
    console.log("Error parsing JSON string:", err);
  }
});

var server = app.listen(PORT, function() {
    console.log('Server listening on ' + PORT);
});


app.get('/points', function(req, res) {
    console.log('get points');
    res.send({ points: points });
});

function saveInfoAndCloseServer(){
    const jsonString = JSON.stringify(points);
    fs.writeFile('./points.json', jsonString, err => {
        if (err) {
            console.log('Error writing file', err)
        } else {
            console.log('Successfully wrote file')
        }
    })

    
    console.log('Closing http server.');
    server.close(() => {
        console.log('Http server closed.');
    });
}

// process.on('SIGTERM', () => {
//     console.info('SIGTERM signal received.');
//     saveInfoAndCloseServer();
// });

process.on('SIGINT', () => {
    console.info('SIGINT signal received.');
    saveInfoAndCloseServer();
});


app.post('/points', function(req, res) {
    points.push({
        key: req.body.key,
        x: parseInt(req.body.x),
        y: parseInt(req.body.y),
        z: parseInt(req.body.z)
    });

    res.send('Successfully created point!');
});

app.put('/points/:key', function(req, res) {
    var key = req.params.key;
    var newX = req.body.newX;
    var newY = req.body.newY;
    var newZ = req.body.newZ;

    var found = false;

    points.forEach(function(point, index) {
        if (!found && point.key === key) {
            point.x = parseInt(newX);
            point.y = parseInt(newY);
            point.z = parseInt(newZ);
        }
    });

    res.send('Succesfully updated point!');
});

app.delete('/points/:key', function(req, res) {
    var key = req.params.key;

    var found = false;

    points.forEach(function(point, index) {
        if (!found && point.key === key) {
            points.splice(index, 1);
        }
    });

    res.send('Successfully deleted point!');
});

