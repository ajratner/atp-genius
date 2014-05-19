from django.conf.urls import patterns, url
from predict import views

urlpatterns = patterns('',
  url(r'^$', views.index, name='index'),
  url(r'^get_prediction/$', views.get_prediction, name='get_prediction')
)
