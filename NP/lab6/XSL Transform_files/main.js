(function() {
  var $, analyzeResult, autotransform, doneTypingInterval, htmlResult, htmlToIframe, pdfResult, pdfToIframe, plainResult, reset, resultButton, resulteditor, transform, typingTimer, updateEngine, xmleditor, xsleditor;

  $ = jQuery;

  doneTypingInterval = 1000;

  typingTimer = null;

  autotransform = true;

  analyzeResult = function(data) {
    var htmlRegex, pdfNamespace, pdfRoot, xmlRegex;
    htmlRegex = XRegExp('^<!DOCTYPE (HTML|html).*?>', "s");
    xmlRegex = XRegExp('<\?xml version="1\.0" encoding=".*?"\?>');
    if (htmlRegex.test(data)) {
      return htmlResult();
    } else if (xmlRegex.test(data)) {
      pdfNamespace = XRegExp('xmlns:(.*?)="http://www\.w3\.org/1999/XSL/Format"');
      if (pdfNamespace.test(data)) {
        pdfRoot = new XRegExp("<" + XRegExp.exec(data, pdfNamespace)[1] + ":root .*?>", "s");
        if (pdfRoot.test(data)) {
          return pdfResult();
        } else {
          return plainResult();
        }
      } else {
        return plainResult();
      }
    } else {
      return plainResult();
    }
  };

  pdfResult = function() {
    resultButton("PDF");
    if ($('#resultview:visible').length > 0) {
      return pdfToIframe();
    }
  };

  htmlResult = function() {
    resultButton("HTML");
    if ($('#resultview:visible').length > 0) {
      return htmlToIframe();
    }
  };

  plainResult = function() {
    $('#viewButton').remove();
    $('#resultview').hide();
    return $('#resulteditor').show();
  };

  pdfToIframe = function() {
    return $('#files').submit();
  };

  htmlToIframe = function() {
    var iframe, iframedoc;
    iframe = $('#resultview iframe').get(0);
    iframedoc = iframe.document;
    if (iframe.contentDocument) {
      iframedoc = iframe.contentDocument;
    } else if (iframe.contentWindow) {
      iframedoc = iframe.contentWindow.document;
    }
    iframedoc.open();
    iframedoc.writeln($('#result').val());
    return iframedoc.close();
  };

  resultButton = function(resultType) {
    $('#viewButton').remove();
    $('#result-window').append("<div id=\"viewButton\" class=\"label-view\" data-result-type=\"" + resultType + "\">" + resultType + "</div>");
    if ($('#resultview:visible').length > 0) {
      return $('#viewButton').addClass('active');
    }
  };

  $('#result-window').on('click', '#viewButton', function() {
    $('#resulteditor').hide();
    $('#resultview').show();
    $('#editorButton').removeClass('active');
    $('#viewButton').addClass('active');
    switch ($(this).data('result-type')) {
      case 'PDF':
        return pdfToIframe();
      case 'HTML':
        return htmlToIframe();
    }
  });

  $('#result-window').on('click', '#editorButton', function() {
    $('#resultview').hide();
    $('#resulteditor').show();
    $('#editorButton').addClass('active');
    return $('#viewButton').removeClass('active');
  });

  transform = function() {
    return jsRoutes.controllers.Application.transform().ajax({
      data: $("#files").serialize(),
      success: function(data) {
        resulteditor.getSession().setValue(data);
        $("#result").val(resulteditor.getSession().getValue());
        return analyzeResult(data);
      },
      error: function(err) {
        return console.log("error: " + err);
      }
    });
  };

  xmleditor = ace.edit("xmleditor");

  xmleditor.setTheme("ace/theme/monokai");

  xmleditor.getSession().setMode("ace/mode/xml");

  xmleditor.getSession().setValue($("xml").val());

  xmleditor.getSession().on('change', function() {
    $("#xml").val(xmleditor.getSession().getValue());
    if (autotransform) {
      clearTimeout(typingTimer);
      return typingTimer = setTimeout(transform, doneTypingInterval);
    }
  });

  xsleditor = ace.edit("xsleditor");

  xsleditor.setTheme("ace/theme/monokai");

  xsleditor.getSession().setMode("ace/mode/xml");

  xsleditor.getSession().setValue($("xsl").val());

  xsleditor.getSession().on('change', function() {
    $("#xsl").val(xsleditor.getSession().getValue());
    if (autotransform) {
      clearTimeout(typingTimer);
      return typingTimer = setTimeout(transform, doneTypingInterval);
    }
  });

  resulteditor = ace.edit("resulteditor");

  resulteditor.setTheme("ace/theme/monokai");

  resulteditor.getSession().setMode("ace/mode/xml");

  $("#autotransform").click(function() {
    autotransform = !autotransform;
    $("#autotransform").find("i").toggleClass("fa-check-square-o");
    return $("#autotransform").find("i").toggleClass("fa-square-o");
  });

  $("#new").click(function() {
    return reset();
  });

  $("#transform").click(function() {
    if (!autotransform) {
      return transform();
    }
  });

  $("#save").click(function() {
    return jsRoutes.controllers.Application.save().ajax({
      data: $("#files").serialize(),
      success: function(data) {
        if (data[1] === "0") {
          window.history.pushState("", "XSL Transform", "/" + data[0]);
        } else {
          window.history.pushState("", "XSL Transform", "/" + data[0] + "/" + data[1]);
        }
        $("#id_slug").val(data[0]);
        return $("#save").find("span").html("Update");
      },
      error: function(err) {
        console.log(err);
        $("#alert").find("h4").html("Save oeps");
        $("#alert").find("p").html(err.responseText);
        $("#alert").alert();
        $("#alert").slideDown();
        return setTimeout(function() {
          return $('#alert').slideUp();
        }, 4000);
      }
    });
  });

  $("#pdf").click(function() {
    $("#files").get(0).target = "_blank";
    $("#files").get(0).action = jsRoutes.controllers.Application.pdf().url;
    return $("#files").get(0).submit();
  });

  $("#engines a").click(function() {
    engine = $(this).data('engine');;
    updateEngine();
    return transform();
  });

  updateEngine = function() {
    $('#engine').val(engine);
    return $('#engine-dropdown').text($('a[data-engine="' + engine + '"]').text());
  };

  reset = function() {
    $("#save").find("span").html("Save");
    $("#id_slug").val("");
    jsRoutes.controllers.Application.defaultXML().ajax({
      dataType: 'text',
      success: function(data) {
        xmleditor.getSession().setValue(data);
        return $("#xml").val(data);
      }
    });
    jsRoutes.controllers.Application.defaultXSL().ajax({
      dataType: 'text',
      success: function(data) {
        xsleditor.getSession().setValue(data);
        return $("#xsl").val(data);
      }
    });
    return window.history.pushState("", "XSL Transform", "/");
  };

  $(function() {
    var clip;
    if (id !== "") {
      $("#save").find("span").html("Update");
      jsRoutes.controllers.Application.xml(id, revision).ajax({
        dataType: 'text',
        success: function(data) {
          xmleditor.getSession().setValue(data);
          return $("#xml").val(data);
        }
      });
      jsRoutes.controllers.Application.xsl(id, revision).ajax({
        dataType: 'text',
        success: function(data) {
          xsleditor.getSession().setValue(data);
          return $("#xsl").val(data);
        }
      });
    } else {
      reset();
    }
    clip = new ZeroClipboard($("#copyclipboard"), {
      moviePath: "/assets/flash/ZeroClipboard.swf"
    });
    clip.glue($("#copyclipboard"));
    return updateEngine();
  });

  $("#copyclipboard").on('mouseover', function(event) {
    return $("#copyclipboard").attr("data-clipboard-text", "" + window.location);
  });

}).call(this);
