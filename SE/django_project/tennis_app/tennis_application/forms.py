from django import forms
from tennis_application.models import Player, Sponsor, Tournament

class PlayerForm(forms.ModelForm):

    name     = forms.CharField(max_length=15, help_text = "Имя")
    family   = forms.CharField(max_length=30, help_text = "Фамилия")
    birthday = forms.DateTimeField(help_text = "Дата рождения")
    count_wons = forms.PositiveIntegerField(help_text = "Количество побед")
    sex = forms.IntegerField(blank=True, null=True, choices=SEX_CHOICES, help_text = "Пол")

    class Meta:
        model = Player
        fields = ('name', 'family', 'birthday', 'count_wons', 'sex',)


class SponsorForm(forms.ModelForm):
    
    name = forms.CharField(max_length=30, help_text = "Название организации")
    address = forms.CharField(max_length=50, help_text = "Адрес")
    players = forms.ModelMultipleChoiceField(queryset=Player.objects.all(), widget=forms.CheckboxSelectMultiple)
        # courses = forms.ModelMultipleChoiceField(queryset=Course.objects.all(),
        #                                widget=FilteredSelectMultiple('Courses', False),
        #                                required=False)

    class Meta:
        model = Sponsor
        fields = ('name', 'address', 'players',)
        

class TournamentForm(forms.ModelForm):

    name      = forms.CharField(max_length=50, help_text = "Название турнира2")
    begin_tur = forms.DateTimeField(help_text = "Начало")
    end_tur   = forms.DateTimeField(help_text = "Окончание")
    category  = forms.IntegerField(blank=True, null=True, choices=CATEGORIES_TOUR, help_text = "Категория")
    sponsors  = forms.ModelMultipleChoiceField(queryset=Sponsor.objects.all(), widget=forms.CheckboxSelectMultiple)
    players   = forms.ModelMultipleChoiceField(queryset=Player.objects.all(), widget=forms.CheckboxSelectMultiple)

    class Meta:
        model = Tournament
        fields = ('name', 'begin_tur', 'end_tur', 'category')

    def clean_name(self):
        data = self.cleaned_data['name']
        if set(data) & set('0123456789') != set():
            raise forms.ValidationError("В названии турнира не должно быть цифр")

        # Always return a value to use as the new cleaned data, even if
        # this method didn't change it.
        return data