# import lab3
# import lab3.analysis.basic as lab
import lab3.experiment as experiment
# import lab3.signal as signal
import inspect

# we should only include classes here that subclass Item...?

experiment_classes = dict(inspect.getmembers(experiment.base, inspect.isclass))
# event_classes = dict(inspect.getmembers(signal.events, inspect.isclass))
group_classes = dict(inspect.getmembers(experiment.group, inspect.isclass))

REGISTRY = {}
REGISTRY.update(experiment_classes)
# REGISTRY.update(event_classes)
REGISTRY.update(group_classes)


def lookup_class(name):

    try:
        return REGISTRY[name]
    except Exception:
        return REGISTRY['Item']
