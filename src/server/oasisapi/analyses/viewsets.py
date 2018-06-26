from __future__ import absolute_import

from django.utils.translation import ugettext_lazy as _
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters import rest_framework as filters

from ..analysis_models.models import AnalysisModel
from ..filters import TimeStampedFilter, CsvMultipleChoiceFilter, CsvModelMultipleChoiceFilter
from ..files.views import handle_related_file
from .models import Analysis
from .serializers import AnalysisSerializer


class AnalysisFilter(TimeStampedFilter):
    name = filters.CharFilter(
        help_text=_('Filter results by case insensitive names equal to the given string'),
        lookup_expr='iexact'
    )
    name__contains = filters.CharFilter(
        help_text=_('Filter results by case insensitive name containing the given string'),
        lookup_expr='icontains',
        field_name='name'
    )
    status = filters.ChoiceFilter(
        help_text=_('Filter results by results in the current analysis status, one of [{}]'.format(
            ', '.join(Analysis.status_choices._db_values))
        ),
        choices=Analysis.status_choices,
    )
    status__in = CsvMultipleChoiceFilter(
        help_text=_(
            'Filter results by results where the current analysis status '
            'is one of a given set (provide multiple parameters or comma separated list), '
            'from [{}]'.format(', '.join(Analysis.status_choices._db_values))
        ),
        choices=Analysis.status_choices,
        field_name='status',
        label=_('Status in')
    )
    model = filters.ModelChoiceFilter(
        help_text=_('Filter results by the id of the model the analysis belongs to'),
        queryset=AnalysisModel.objects.all(),
    )
    model__in = CsvModelMultipleChoiceFilter(
        help_text=_('Filter results by the id of the model the analysis belongs to'),
        field_name='model',
        label=_('Model in'),
        queryset=AnalysisModel.objects.all(),
    )

    class Meta:
        model = Analysis
        fields = [
            'name',
            'name__contains',
            'status',
            'status__in',
            'model',
            'model__in',
        ]
        
    def __init__(self, *args, **kwargs):
        super(AnalysisFilter, self).__init__(*args, **kwargs)
        


class AnalysisViewSet(viewsets.ModelViewSet):
    """
    list:
    Returns a list of Analysis objects.

    ### Examples

    To get all analyses with 'foo' in their name

        /analyses/?name__contains=foo

    To get all analyses created on 1970-01-01

        /analyses/?created__date=1970-01-01

    To get all analyses updated before 2000-01-01

        /analyses/?modified__lt=2000-01-01

    To get all analyses in the `NOT_RAN` state

        /analyses/?status=NOT_RAN

    To get all started and pending tasks

        /analyses/?status__in=PENDING&status__in=STARTED

    To get all models in model `1`

        /analyses/?model=1

    To get all models in models `2` and `3`

        /analyses/?model__in=2&model__in=3

    retrieve:
    Returns the specific analysis entry.

    create:
    Creates a analysis based on the input data

    update:
    Updates the specified analysis

    partial_update:
    Partially updates the specified analysis (only provided fields are updated)
    """
    
    queryset = Analysis.objects.all()
    serializer_class = AnalysisSerializer
    filter_class = AnalysisFilter

    @action(methods=['post'], detail=True)
    def run(self, request, pk=None):
        """
        Runs all the analysis. The analysis must have one of the following
        statuses, `NOT_RAN`, `STOPPED_COMPLETED`, `STOPPED_CANCELLED` or
        `STOPPED_ERROR`
        """
        obj = self.get_object()
        obj.run(request)
        return Response(self.get_serializer(instance=obj, context=self.get_serializer_context()).data)

    @action(methods=['post'], detail=True)
    def cancel(self, request, pk=None):
        """
        Cancels a currently running analysis. The analysis must have one of the following
        statuses, `PENDING` or `STARTED`
        """
        obj = self.get_object()
        obj.cancel()
        return Response(self.get_serializer(instance=obj, context=self.get_serializer_context()).data)

    @action(methods=['post'], detail=True)
    def copy(self, request, pk=None):
        """
        Copies an existing analysis, copying the associated input files and model and modifying
        it's name (if none is provided) and resets the status, input errors and outputs
        """
        obj = self.get_object()
        new_obj = obj.copy()

        serializer = self.get_serializer(instance=new_obj, data=request.data, context=self.get_serializer_context(), partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data)

    @action(methods=['get', 'post', 'delete'], detail=True)
    def settings_file(self, request, pk=None):
        """
        get:
        Gets the portfolios `settings_file` contents

        post:
        Sets the portfolios `settings_file` contents

        delete:
        Disassociates the portfolios `settings_file` contents
        """
        return handle_related_file(self.get_object(), 'settings_file', request, ['application/json'])

    @action(methods=['get', 'post', 'delete'], detail=True)
    def input_file(self, request, pk=None):
        """
        get:
        Gets the portfolios `input_file` contents

        post:
        Sets the portfolios `input_file` contents

        delete:
        Disassociates the portfolios `input_file` contents
        """
        return handle_related_file(self.get_object(), 'input_file', request, ['application/x-gzip', 'application/gzip', 'application/x-tar', 'application/tar'])

    @action(methods=['get', 'delete'], detail=True)
    def input_errors_file(self, request, pk=None):
        """
        get:
        Gets the portfolios `input_errors_file` contents

        post:
        Sets the portfolios `input_errors_file` contents

        delete:
        Disassociates the portfolios `input_errors_file` contents
        """
        return handle_related_file(self.get_object(), 'input_errors_file', request, ['application/json', 'text/csv'])

    @action(methods=['get', 'delete'], detail=True)
    def output_file(self, request, pk=None):
        """
        get:
        Gets the portfolios `output_file` contents

        post:
        Sets the portfolios `output_file` contents

        delete:
        Disassociates the portfolios `output_file` contents
        """
        return handle_related_file(self.get_object(), 'output_file', request, ['application/x-gzip', 'application/gzip', 'application/x-tar', 'application/tar'])
