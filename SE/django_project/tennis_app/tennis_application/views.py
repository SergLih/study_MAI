from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse, Http404
from django.core import serializers
from tennis_application.models import Player, Sponsor, Tournament
from django.core.files.storage import FileSystemStorage
from django.views.decorators.http import *

from rest_framework import viewsets, parsers
from rest_framework import permissions
from tennis_application.serializes import PlayerSerializer, SponsorSerializer, TournamentSerializer

@require_POST
def create_user(request):
    return JsonResponse({"id": 1})

@require_GET
def read_user(request, user_id):
    return JsonResponse({"id": user_id})

@require_GET
def show_player(request, id):
    try:
        qplayer_json = serializers.serialize('json', Player.objects.filter(player_id=id))
    except Player.DoesNotExist:
        raise Http404("Игрок не найден")
    return HttpResponse(qplayer_json, content_type='application/json')

@require_GET
def get_all_players(request):
    qplayers = Player.objects.all()
    qplayers_json = serializers.serialize('json', qplayers)
    return HttpResponse(qplayers_json, content_type='application/json')

@require_GET
def hello(request):
    msg = "Welcome: tennis application"
    return HttpResponse(msg, content_type='text/plain')

@require_GET
def show_surface(request, surface_slug):
    if surface_slug == 'clay':   
        return JsonResponse({"surface": surface_slug, "players": [{"name": "Nadal"}, {"name": "Borg"}]})
    elif surface_slug == 'hard':
        return JsonResponse({"surface": surface_slug, "players": [{"name": "Djokovich"}, {"name": "Medvedev"}]})

@require_GET
def show_surface_player(request, surface_slug, player_name_slug):
    return JsonResponse({"surface": surface_slug, "player": [{"name": player_name_slug}]})

@require_GET
def get_my_ip(request):
    return JsonResponse({
        'ip': request.META.get('HTTP_X_REAL_IP') or request.META.get('REMOTE_ADDR'),
    })

class PlayerViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = Player.objects.all()
    serializer_class = PlayerSerializer
    permission_classes = [permissions.IsAuthenticated]

class SponsorViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = Sponsor.objects.all()
    serializer_class = SponsorSerializer
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    permission_classes = [permissions.IsAuthenticated]

class TournamentViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = Tournament.objects.all()
    serializer_class = TournamentSerializer
    permission_classes = [permissions.IsAuthenticated]


def image_upload(request):
    if request.method == 'POST':
        image_file = request.FILES['image_file']
        image_type = request.POST['image_type']
        if settings.USE_S3:
            upload = Sponsor(logo=image_file)
            upload.save()
            image_url = upload.file.url
        else:
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            image_url = fs.url(filename)
        return render(request, {
            'image_url': image_url
        })
    return request