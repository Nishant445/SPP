

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Feedback',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('email', models.EmailField(max_length=254)),
                ('satisfaction', models.CharField(choices=[('very_satisfied', 'Very Satisfied'), ('satisfied', 'Satisfied'), ('neutral', 'Neutral'), ('dissatisfied', 'Dissatisfied'), ('very_dissatisfied', 'Very Dissatisfied')], max_length=20)),
                ('accuracy', models.CharField(choices=[('very_accurate', 'Very Accurate'), ('accurate', 'Accurate'), ('neutral', 'Neutral'), ('inaccurate', 'Inaccurate'), ('very_inaccurate', 'Very Inaccurate')], max_length=20)),
                ('improvements', models.TextField()),
                ('additional_feedback', models.TextField(blank=True)),
            ],
        ),
    ]
