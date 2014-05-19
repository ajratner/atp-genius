var page = require('webpage').create();
page.open('http://www.tennisabstract.com/cgi-bin/player.cgi?p=AlexKuznetsov&f=ACareerqqs31o1', function() {
  console.log(page.content);
  phantom.exit();
});
