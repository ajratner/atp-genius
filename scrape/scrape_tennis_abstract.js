var page = require('webpage').create();
page.open('http://tennisabstract.com/cgi-bin/player.cgi?p=TaroDaniel&f=o1', function() {
  console.log(page.content);
  phantom.exit();
});
