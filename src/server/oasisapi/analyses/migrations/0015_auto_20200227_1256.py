# Generated by Django 2.2.8 on 2020-02-27 12:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analyses', '0014_auto_20200220_1258'),
    ]

    operations = [
        migrations.AddField(
            model_name='analysistaskstatus',
            name='pending_time',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AlterField(
            model_name='analysistaskstatus',
            name='queue_time',
            field=models.DateTimeField(default=None, editable=False, null=True),
        ),
    ]
