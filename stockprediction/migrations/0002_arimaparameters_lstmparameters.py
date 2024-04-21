

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockprediction', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ARIMAParameters',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('p', models.IntegerField(default=1)),
                ('d', models.IntegerField(default=1)),
                ('q', models.IntegerField(default=1)),
            ],
        ),
        migrations.CreateModel(
            name='LSTMParameters',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lstm_units', models.IntegerField(default=50)),
                ('dropout_rate', models.FloatField(default=0.2)),
                ('epochs', models.IntegerField(default=10)),
                ('batch_size', models.IntegerField(default=20)),
            ],
        ),
    ]
