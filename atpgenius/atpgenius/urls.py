from django.conf.urls import patterns, include, url
from predict import views

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    url(r'^', include('predict.urls')),
    # Examples:
    # url(r'^$', 'atpgenius.views.home', name='home'),
    # url(r'^atpgenius/', include('atpgenius.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
)
