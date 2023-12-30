var express = require('express');
var app = express();
var bodyParser = require('body-parser');
const fs = require('fs');
const mysql = require("mysql2");

var points = [];

var PORT = process.env.PORT || 3000;

app.use(express.static(__dirname));
app.use(bodyParser.json());

const connection = mysql.createConnection({
      host: "127.0.0.1",
      user: "admin",
      database: "pointsdb",
      password: "Sergiomysql555!"
    });

connection.connect(function(err){
        if (err) {
          return console.error("Ошибка: " + err.message);
        }
        else{
          console.log("Подключение к серверу MySQL успешно установлено");
        }
     });

var points = [];
//fs.readFile("./points.json", "utf8", (err, jsonString) => {
connection.execute("SELECT * FROM points",
  function(err, results, fields) {
    console.log(err);
    console.log(results); // собственно данные
    results.forEach(function(row, index){
        points.push({
            key: row.name,
            x: row.x,
            y: row.y,
            z: row.z
        });
    });
});
  

var server = app.listen(PORT, function() {
    console.log('Server listening on ' + PORT);
});


app.get('/points', function(req, res) {
    console.log('get points');
    res.send({ points: points });
});

function saveInfoAndCloseServer(){
    

    

    connection.query("DELETE FROM points",
      function(err, results) {
            if(err) console.log(err);
            else console.log("Старые данные удалены");
    });

    points.forEach(function(pt, index) {
        const sql = "INSERT INTO points(id, name, x, y, z) VALUES(?, ?, ?, ?, ?)";
        const ptdata = [index+1, pt.key, pt.x, pt.y, pt.z];
        connection.query(sql, ptdata, function(err, results) {
            if(err) console.log(err);
            else console.log("Данные добавлены: ", pt);
        });
    });



     // закрытие подключения
    connection.end(function(err) {
      if (err) {
        return console.log("Ошибка: " + err.message);
      }
      console.log("Подключение закрыто");
    });
    
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

