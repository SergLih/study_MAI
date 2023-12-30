from django.conf.urls import url
from django.urls import path
from tennis_application import views

urlpatterns = [
    path('user/<int:user_id>/', views.read_user, name='read_user'),
	url(r'^create/', views.create_user, name='create_user'),
    url(r'^players/$', views.get_all_players, name='get_all_players'),
    url(r'^players/(?P<id>\d+)/$', views.show_player, name='show_player'),
    # url(r'^(?P<surface_slug>[\w\-]+)/$', views.show_surface, name='show_surface'),
    # url(r'^(?P<surface_slug>[\w\-]+)/(?P<player_name_slug>[\w\-]+)/$', views.show_surface_player, name='show_surface_player'),

]