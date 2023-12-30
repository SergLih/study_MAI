$(function() {
    // GET/READ
    $('#get-button').on('click', function() {
        $.ajax({
            url: '/points',
            contentType: 'application/json',
            success: function(response) {
                var tbodyEl = $('tbody');
                console.log(response.points);
                tbodyEl.html('');

                response.points.forEach(function(point) {
                    tbodyEl.append('\
                        <tr>\
                            <td class="key">' + point.key+ '</td>\
                            <td><input type="text" class="x" value="' + point.x + '"></td>\
                            <td><input type="text" class="y" value="' + point.y + '"></td>\
                            <td><input type="text" class="z" value="' + point.z + '"></td>\
                            <td>\
                                <button class="update-button">🖊</button>\
                                <button class="delete-button">❌</button>\
                            </td>\
                        </tr>\
                    ');
                });
            }
        });
    });

    // CREATE/POST
    $('#create-form').on('submit', function(event) {
        event.preventDefault();

        //var createInput = $('#create-input')
        // var createKey = $('#key');
        // var createX = $('#x');
        // var createY = $('#y');
        // var createZ = $('#z');

        $.ajax({
            url: '/points',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ 
                key: $('#key').val(),
                x:   $('#x').val(),
                y:   $('#y').val(),
                z:   $('#z').val()
            }),
            success: function(response) {
                console.log(response);
                $('#key').val('');
                $('#x').val('');
                $('#y').val('');
                $('#z').val('');
                $('#get-button').click();
            }
        });
    });

    // UPDATE/PUT
    $('table').on('click', '.update-button', function() {
        var rowEl = $(this).closest('tr');
        var key = rowEl.find('.key').text();
        var newX = rowEl.find('.x').val();
        var newY = rowEl.find('.y').val();
        var newZ = rowEl.find('.z').val();

        $.ajax({
            url: '/points/' + key,
            method: 'PUT',
            contentType: 'application/json',
            data: JSON.stringify({ newX: newX, newY: newY, newZ: newZ }),
            success: function(response) {
                console.log(response);
                $('#get-button').click();
            }
        });
    });

    // DELETE
    $('table').on('click', '.delete-button', function() {
        var rowEl = $(this).closest('tr');
        var key = rowEl.find('.key').text();

        $.ajax({
            url: '/points/' + key,
            method: 'DELETE',
            contentType: 'application/json',
            success: function(response) {
                console.log(response);
                $('#get-button').click();
            }
        });
    });
});
