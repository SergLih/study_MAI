from rest_framework import serializers
from tennis_application.models import Player, Sponsor, Tournament
from rest_framework.renderers import JSONRenderer

class PlayerSerializer(serializers.HyperlinkedModelSerializer):
    
    # serializers.PrimaryKeyRelatedField(many=True)
    
    class Meta:
        model = Player
        fields = ['url', 'name', 'family', 'birthday', 'count_wons', 'sex']
        # extra_kwargs = {
        #     'url': {'view_name': 'player-detail', 'lookup_field': 'id'},
        # }
        
class SponsorSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Sponsor
        fields = ['url', 'logo', 'name', 'address', 'players']
        # extra_kwargs = {
        #     'url': {'view_name': 'event-detail', 'lookup_field': 'id'},
        # }

class TournamentSerializer(serializers.HyperlinkedModelSerializer):
    
    # players = PlayerSerializer(many=True)
    players = PlayerSerializer(many=True)
    players = JSONRenderer().render(players.data)

    sponsors = SponsorSerializer(many=True)
    sponsors = JSONRenderer().render(sponsors.data)


    class Meta:
        model = Tournament
        fields = ['url', 'name', 'begin_tur', 'end_tur', 'category', 'sponsors', 'players']