"""tennis_app URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path
from django.conf.urls import url
from tennis_application import views
from django.conf.urls import include
from rest_framework import routers
from django.conf import settings  # new
from django.conf.urls.static import static

router = routers.DefaultRouter()
router.register(r'players', views.PlayerViewSet)
router.register(r'sponsors', views.SponsorViewSet)
router.register(r'tournaments', views.TournamentViewSet)

urlpatterns = [
    url(r'^$', views.hello, name='hello'),
    path('api/', include(router.urls)),
    # path('', include('social_django.urls')),
    path('api/admin/', include('rest_framework.urls', namespace='rest_framework'))
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# urlpatterns = [
#     url(r'^$', views.hello, name='hello'),
#     url(r'^api/admin/', admin.site.urls),
#     url(r'^api/tennis/', include('tennis_application.urls')),
# ] + staticfiles_urlpatterns()
