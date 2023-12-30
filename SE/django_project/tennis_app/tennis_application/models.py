from django.db import models
from django.core.exceptions import ValidationError

SEX_CHOICES = [
    (0, 'Мужчина'),
    (1, 'Женщина')
]

CATEGORIES_TOUR = [
    (0, 'I'),
    (1, 'II'),
    (2, 'III'),
    (3, 'IV'),
]

class Player(models.Model):
	player_id = models.AutoField(primary_key=True, verbose_name = "Id")
	name     = models.CharField(max_length=15, verbose_name = "Имя")
	family   = models.CharField(max_length=30, verbose_name = "Фамилия")
	birthday = models.DateTimeField(verbose_name = "Дата рождения")
	count_wons = models.PositiveIntegerField(verbose_name = "Количество побед")
	sex = models.IntegerField(blank=True, null=True, choices=SEX_CHOICES, verbose_name = "Пол")

	def __str__(self):
		return f'{self.family} {self.name}'

	def save(self, *args, **kwargs):
		results = Player.objects.filter(name=self.name, family=self.family)
		for o1 in results:
			if o1.player_id != self.player_id:
				raise ValidationError('Игрок с таким именем уже есть')

		super().save(*args, **kwargs)

class Sponsor(models.Model):
	sponsor_id = models.AutoField(primary_key=True, verbose_name = "Id")
	logo = models.FileField(max_length=100, null=True)
	name = models.CharField(max_length=30, verbose_name = "Название организации")
	address = models.CharField(max_length=50, verbose_name = "Адрес")
	players = models.ManyToManyField(Player)

	def __str__(self):
		return self.name

class Tournament(models.Model):
	tournament_id = models.AutoField(primary_key=True, verbose_name = "Id")
	name      = models.CharField(max_length=50, verbose_name = "Название турнира")
	begin_tur = models.DateTimeField(verbose_name = "Начало")
	end_tur   = models.DateTimeField(verbose_name = "Окончание")
	category  = models.IntegerField(blank=True, null=True, choices=CATEGORIES_TOUR, verbose_name = "Категория")
	sponsors  = models.ManyToManyField(Sponsor)
	players   = models.ManyToManyField(Player)

	def __str__(self):
		return self.name

	def _validate_start_end_dates(self):
		if self.end_tur < self.begin_tur:
			raise ValidationError("Дата окончания не может быть раньше даты начала")
    
	def save(self, *args, **kwargs):
		self._validate_start_end_dates()
		return super().save(*args, **kwargs)
