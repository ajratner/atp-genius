
// AJAX UTILITY FUNCTIONS

// utility function to get cookie
function getCookie(name) {
  var cookieValue = null;
  if (document.cookie && document.cookie != '') {
    var cookies = document.cookie.split(';');
    for (var i=0; i < cookies.length; i++) {
      var cookie = $.trim(cookies[i]);
      if (cookie.substring(0, name.length + 1) == (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

// utility function to determine if method needs CSRF protection
function csrfSafeMethod(method) {
  return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

var test = '';

$(document).ready(function() {

  // load player names in from html
  var PLAYERS = [];
  $("#player-names").children("li").each(function() {
    PLAYERS.push($(this).html());
  });

  //var PLAYERS = ['Alex', 'Bob', 'Jim', 'Joe'];

  // constructs the suggestion engine
  var players = new Bloodhound({
    datumTokenizer: Bloodhound.tokenizers.obj.whitespace('value'),
    queryTokenizer: Bloodhound.tokenizers.whitespace,
    local: $.map(PLAYERS, function(player) { return { value: player }; })
  });
   
  // kicks off the loading/processing of `local` and `prefetch`
  players.initialize();
   
  $('.typeahead').typeahead({
    hint: true,
    highlight: true,
    minLength: 1
  },
  {
    name: 'players',
    displayKey: 'value',
    // `ttAdapter` wraps the suggestion engine in an adapter that
    // is compatible with the typeahead jQuery plugin
    source: players.ttAdapter()
  });


  // clicking the predict button- getting the prediction via ajax
  function predict() {
    $("#prediction-results").hide();
    $("#prediction-loading").show();
    
    // send query 
    $.ajax({ 
      url: '/get_prediction/',
      type: 'POST',
      crossDomain: false,
      beforeSend: function(xhr, settings) {
        if (!csrfSafeMethod(settings.type)) {
          xhr.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
        }
      },
      data: {
        'p1-name' : $("#p1-name").val(),
        'p2-name' : $("#p2-name").val(),
        'surface' : $("#surface").val(),
        'tournament-round' : $("#tournament-round").val()
      },
      success: function(data) {
        $("#predicted-winner").html(data['winner']);
        $("#predicted-confidence").html(data['margin']);
        $(".fv").each(function(i) {
          $(this).html(data['feature-vector'][i]);
        });
        $("#prediction-loading").hide();
        $("#prediction-results").show();
      }
    });
  }

  $("#predict-button").click(function() { predict(); });

});
