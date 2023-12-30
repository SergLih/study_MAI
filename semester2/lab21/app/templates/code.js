var cy;


$(function(){

  cy = cytoscape({
    
                    container: document.getElementById('cy'),
                    ready: function(){
                    },
                    layout: {
                        name: 'grid',
                        rows: 2,
                        cols: 2
                    },
                    style: [
                        {
                            selector: 'node',
              style: {
                shape: 'round',
                'height': 50,
        'width': 50,
        'background-fit': 'cover',
        'border-color': '#000',
        'border-width': 3,
        'border-opacity': 0.5,
                label: 'data(id)',
                'text-valign': 'center',
            'color': 'white',
            'text-outline-width': 1,
            'text-outline-color': 'black',
            'background-color': 'gray'
                            }
                        },
                        {
                            selector: '.edgehandles-hover',
                            css: {
                                'background-color': 'red'
                            }
                        },
                        {
                            selector: '.edgehandles-source',
                            css: {
                                'border-width': 2,
                                'border-color': 'red'
                            }
                        },
                        {
                            selector: '.edgehandles-target',
                            css: {
                                'border-width': 2,
                                'border-color': 'red'
                            }
                        },
                        {
                            selector: '.edgehandles-preview, .edgehandles-ghost-edge',
                            css: {
                                'line-color': 'red',
                                'target-arrow-color': 'red',
                                'source-arrow-color': 'red'
                            }
                        },
                        {
                            selector: '._808080',
                            css: {
                                'background-color': '#808080',
                            }
                        },
                        {
                            selector: '._C0C0C0',
                            css: {
                                'background-color': '#C0C0C0',
                            }
                        },
                        {
                            selector: '._FF00FF',
                            css: {
                                'background-color': '#FF00FF',
                            }
                        },
                        {
                            selector: '._800080',
                            css: {
                                'background-color': '#800080',
                            }
                        },
                        {
                            selector: '._FF0000',
                            css: {
                                'background-color': '#FF0000',
                            }
                        },
                        {
                            selector: '._FFFF00',
                            css: {
                                'background-color': '#FFFF00',
                            }
                        },
                        {
                            selector: '._800000',
                            css: {
                                'background-color': '#800000',
                            }
                        },
                        {
                            selector: '._808000',
                            css: {
                                'background-color': '#808000',
                            }
                        },
                        {
                            selector: '._00FF00',
                            css: {
                                'background-color': '#00FF00',
                            }
                        },
                        {
                            selector: '._00FFFF',
                            css: {
                                'background-color': '#00FFFF',
                            }
                        },
                        {
                            selector: '._008080',
                            css: {
                                'background-color': '#008080',
                            }
                        },
                        {
                            selector: '._0000FF',
                            css: {
                                'background-color': '#0000FF',
                            }
                        },
                        {
                            selector: '._000080',
                            css: {
                                'background-color': '#000080',
                            }
                        },
                        {
                            selector: '._008000',
                            css: {
                                'background-color': '#008000',
                            }
                        },
                        {
                            selector: '.bird',
                            css: {
                                'background-image': 'https://farm8.staticflickr.com/7272/7633179468_3e19e45a0c_b.jpg',
                            }
                        },
                        {
                            selector: '.edge-clique',
                            css: {
                                'line-color': 'black',
                                'width' : 5
                            }
                        }
                    ]
  });

      cy.edgehandles({
   toggleOffOnLeave: true,
   handleNodes: "node",
   handleSize: 10,
   edgeType: function(){ return 'flat'; }
 });
    var i = 0;
    var del_elem = [];
    cy.on('cxttap', 'node' ,function (evt) {
        cy.remove(cy.$("#" + this.id()));
        del_elem[del_elem.length] = this.id();
    });
    cy.on('cxttap', 'edge' ,function (evt) {
        cy.remove(cy.$("#" + this.id()));
    });
    var color_node = ['_FF0000','_00FF00','_0000FF','_FFFF00', '_FF00FF', '_00FFFF', '_808080', '_C0C0C0', '_800080', '_800000',  '_808000', '_008000', '_008080',  '_000080'];
    $('.cytosc').on('tap', '*',function(e) {

            if(del_elem.length == 0) {
                i += 1;
                index = i;
            } else {
                del_elem = del_elem.sort();
                index = del_elem[0];
                del_elem.splice(0,1);
            }
            cy.add([{
                group : "nodes",

                data : {
                    id : index
                },
                renderedPosition : {
                    x: e.pageX,
                    y: e.pageY
                }
            }]);
//cy.$('#'+index).classes(color_node[0]);
    });
  var $config = $('#config');
  var $btnParam = $('<div class="param"></div>');
  $config.append( $btnParam );
  var $divColorVariants = $('<div class="param"></div>');
  $config.append( $divColorVariants );
  var buttons = [
    {
      label: 'Масштаб',//'<i class="fa fa-eye"></i>',
      id :'button_center'
    },
    {
      label: 'Очистка',//'<i class="fa fa-remove"></i>',
      id :'button_remove'
    },
    {
      label: 'Пуск',//'<i class="fa fa-check"></i>',
      id : 'button_result'
    }
  ];
buttons.forEach( makeButton );
var prev_clique = [];
var prev_clique_edges = [];
var ind = 0;
var allColors;
function makeButton( opts ){
    var $button = $('<button class="btn btn-default",id="' + opts.id + '">'+ opts.label +'</button>');
    
    $btnParam.append( $button );

    $button.on('click', function(){
        if(opts.id == 'button_center') {
        cy.center();
        cy.fit();
  } else if(opts.id == 'button_remove'){
        cy.remove('*');
        i = 0;
        del_elem = [];
  } else if(opts.id =='button_result') {
    var data_for_ajax = JSON.stringify(cy.json(), null, '\t');
    try{
        $.ajax({                    // $ - основной объект библиотеки jQuery
            type: "POST",
            url: '_find',
            data: JSON.stringify(cy.json(), null, '\t'),
            contentType: 'application/json;charset=UTF-8',
            success: function(msg){
                //console.log(data_for_ajax);
                console.log(msg);
                $divColorVariants.empty();
                var $selColorVariants = $('<select id="selColorVariants"></select>');
                $divColorVariants.append($selColorVariants);
                //$divColorVariants.html(allColors);
                jsonData = $.parseJSON(msg);
                allColors = jsonData['allColors']
                vertex_ids = jsonData['vertex_ids']
                console.log(jsonData);
                for (var i=0;i<allColors.length;i++){
                   $('<option/>').val(i).html('Раскраска ' + (i+1).toString()).appendTo('#selColorVariants');
                }
                var colors = allColors[0];      //первая раскраска применяется по умолчанию
                for(var i=0;i<colors.length;i++)
                {
                    cy.$('#'+vertex_ids[i]).classes(color_node[colors[i]]);
                }
                $selColorVariants.on('change', function(){
                    var colorN = Number($("#selColorVariants").val());
                    var colors = allColors[colorN];
                    for(var i=0;i<colors.length;i++)
                    {
                        cy.$('#'+vertex_ids[i]).classes(color_node[colors[i]]);
                    }
                });
                
                return;
                var clique = $.parseJSON(msg);
                for(var i = 0; i < prev_clique.length; i++) {
                    cy.$('#'+prev_clique[i]).classes(color_node[Math.floor(Math.random() * (13 - 0 + 1)) + 0]);
                }
                for(var i = 0; i < prev_clique_edges.length; i++) {
                    cy.$('#'+prev_clique_edges[i]).classes("node");
                }
                prev_clique = [];
                prev_clique_edges = [];
                try {
                    clique[ind].forEach(function(a) {
                        cy.$('#'+a).classes('bird');
                        prev_clique[prev_clique.length] = a;
                    });
                    clique[ind].forEach(function(a) {
                        clique[ind].forEach(function(b) {
                            try{
                                cy.edges("[source = '" + a+ "'][target = '"+b+"']").forEach(function( ele ){
                                    prev_clique_edges[prev_clique_edges.length] = ele.id();
                                    cy.$('#'+ ele.id()).classes("edge-clique");
                            });
                        }catch(err) {}
                        });
                    });
                    ind += 1;
                } catch(err) {
                    ind = 0;
                    clique[0].forEach(function(a) {
                        cy.$('#'+a).classes('bird');
                        prev_clique[prev_clique.length] = a;
                    });
                    clique[ind].forEach(function(a) {
                        clique[ind].forEach(function(b) {
                            try{
                                cy.edges("[source = '" + a+ "'][target = '"+b+"']").forEach(function( ele ){
                                    prev_clique_edges[prev_clique_edges.length] = ele.id();
                                    cy.$('#'+ ele.id()).classes("edge-clique");
                            });
                        }catch(err) {}
                        });
                    });
                    ind += 1;
                }
            }
        });
    } catch(err) {}
  }

      if( opts.fn ){ opts.fn(); }

    });
  }

  cy.nodes().forEach(function(n){
    var g = n.data('name');

    n.qtip({
      position: {
        my: 'top center',
        at: 'bottom center'
      },
      style: {
        classes: 'qtip-bootstrap',
        tip: {
          width: 16,
          height: 8
        }
      }
    });
  });

  $('#config-toggle').on('click', function(){
    $('body').toggleClass('config-closed');
    cy.resize();
  });
});

$(function() {
  FastClick.attach( document.body );
});
