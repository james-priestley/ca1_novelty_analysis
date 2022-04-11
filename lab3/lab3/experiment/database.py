import os.path
import time
import pickle

import MySQLdb
import MySQLdb.cursors
import json


class ExperimentDatabase:
    """An object to wrap MySQLdb and handle all connections to mySQL server

    Example
    -------

    >>> from lab.classes import database
    >>> db = database.ExperimentDatabase()
    >>> db.select_all('SELECT * FROM mice LIMIT 10')
    >>> db.disconnect()
    """

    _trial_fields = ['trial_id', 'mouse_id', 'start_time', 'stop_time',
                     'behavior_file', 'tSeries_path', 'experiment_group',
                     'experiment_id', 'mouse_name']

    def __init__(self):
        """Connects to SQL database upon initialization."""

        self.connect()

    def connect(self):
        """Connect to the SQL database."""

        host = '10.255.0.3'
        user = 'jack'       # TODO: use different credentials
        password = 'flask'  # TODO: update credentials
        database = 'experiments'

        self._database = MySQLdb.connect(
            host=host,
            user=user,
            passwd=password,
            db=database,
            cursorclass=MySQLdb.cursors.DictCursor)

    def disconnect(self):
        """Close the connection to SQL database server."""

        self._database.close()

    def select(self, sql, args=[], verbose=False):
        """ Queries SQL database return single result

        Parameters
        ----------
        sql : str
            Raw SQL to pass to the database.
        args : list, optional
            List of variables to sub into SQL statement in accordiance see
            MySQLdb. Defaults to an empty list.
        verbose : bool, optional
            If set to True print the SQL query. Defaults to False

        Returns
        -------
        result : dict
            First record to match SQL query as a dictionary, with the field
            name being the key.
        """

        cursor = self._database.cursor()
        cursor.execute(sql, args)
        if verbose:
            print(sql)

        result = cursor.fetchone()
        return result

    def select_all(self, sql, args=[], verbose=False):
        """ Queries SQL database and return all the results

        Parameters
        ----------
        sql : str
            Raw SQL to pass to the database.
        args : list, optional
            List of variables to sub into SQL statement in accordiance see
            MySQLdb. Defaults to an empty list.
        verbose : bool, optional
            If set to True print the SQL query. Defaults to False

        Returns
        -------
        result : dict
            All records to match SQL query as a dictionary, with the field name
            being the key.
        """

        cursor = self._database.cursor()
        cursor.execute(sql, args)
        if verbose:
            print(sql)

        result = cursor.fetchall()
        return result

    def query(self, sql, args=[], verbose=False, ignore_duplicates=True):
        """ Run an arbitrary SQL query i.e. INSERT or DELETE commands. SQL
        statement is passsed directory MySQLdb.

        Parameters
        ----------
        sql : str
            Raw SQL statement to query the database with.
        args : list, optional
            List of variables to sub into SQL statement in accordiance see
            MySQLdb. Defaults to an empty list.
        verbose : bool, optional
            If set to True print the SQL query. Defaults to False
        ignore_duplicates : bool, optional
            If set to True Duplicate entry errors are prevented from being
            raised, instead of raising an error the method returns False.
            Defaults to True.

        Returns
        -------
        bool : True if the SQL statement executes, False otherwise
        """

        cursor = self._database.cursor()
        try:
            cursor.execute(sql, args)
        except Exception as e:
            if not ignore_duplicates or 'Duplicate entry' not in e.__str__():
                raise e
            return False
        else:
            if verbose:
                print(sql)
            self._database.commit()
        return True


def add_experiment_to_database(filename, trial_id=None, overwrite=False):
    """Add an experiment to the database from data in pickled behavior file.

    Parameters
    ----------
    filename : str
        Path to pickled behavior data for the experiment to be added
    trial_id : int, optional
        Update this trial_id in database. Defaults to None (a new id is created
        if the behavior file is not associated with an existing database entry)
    overwrite : bool, optional
        Whether re-add experiment to the database, if an existing database
        entry is already associated with this behavior data file
    """
    if not overwrite and fetch_trial_id(behavior_file=filename) is not None:
        print(f'trial {filename} already loaded')
        return fetch_trial_id(behavior_file=filename)

    # get trial info from pickled behavior file
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    try:
        trial_info = data['__trial_info']
    except KeyError:
        raise Exception(f"__trial_info not found in {filename}")

    update_trial(filename, trial_info['mouse'], trial_info['start_time'],
                 trial_info['stop_time'], trial_info['experiment_group'],
                 trial_id=trial_id)

    if trial_id is None:
        trial_id = fetch_trial_id(filename)
    else:
        delete_all_trial_attrs(trial_id)

    print('loading trial {}: {}'.format(trial_id, filename))

    for key in filter(lambda k: k not in [
            'mouse', 'start_time', 'stop_time', 'experiment_group'],
            trial_info.keys()):

        if key == 'contexts':
            for context in trial_info['contexts'].keys():
                update_trial_attr(trial_id, f'{context}_context',
                                  json.dumps(trial_info[key][context]))
        elif isinstance(trial_info[key], dict) or \
                isinstance(trial_info[key], list):

            # TODO : figure out a more general way to handle long entries?
            if key == 'controllers':
                continue
                # for k, v in trial_info[key].items():
                #     update_trial_attr(trial_id, k, json.dumps(v))
            else:
                update_trial_attr(trial_id, key, json.dumps(trial_info[key]))

        else:
            update_trial_attr(trial_id, key, trial_info[key])

    return trial_id


def fetch_mouse_weights(mouse_name):
    """ Fetch all the weight records for a given mouse in the SQL database.

    Parameters
    ----------
    mouse_name : str
        Name of the mouse to search the database for records of weights.

    Returns
    -------
    weights : list
        List of dictionaries with mouse_name, weight, and date keys.
    """

    db = ExperimentDatabase()
    weights = db.select_all("""
        SELECT mouse_name, weight, date
        FROM mice
        INNER JOIN mouse_weights
        ON mice.mouse_id = mouse_weights.mouse_id
        WHERE mouse_name = %s
        ORDER BY date ASC
        """, args=[mouse_name], verbose=False)

    return weights


def read_mouse_weights(mouse_name, limit=10):
    """ Print out a report of recent mouse weight records and the average
    overall weight for the mouse.

    Parameters
    ----------
    mouse_name : str
        Name of the mouse to search database for weight records of.
    limit : int, optional
        Number of records to print. Defaults to 10.
    """

    db = ExperimentDatabase()
    weights = db.select_all("""
        SELECT mouse_name, weight, date
        FROM mice
        INNER JOIN mouse_weights
        ON mice.mouse_id = mouse_weights.mouse_id
        WHERE mouse_name = %s
        ORDER BY date DESC
        LIMIT %s
        """, args=[mouse_name, limit], verbose=False)

    def print_weight(weight):
        print(f"{weight['date']} \t {weight['weight']}")

    print(f'Recent records for mouse {mouse_name}')
    map(print_weight, weights)

    average = db.select("""
        SELECT AVG(weight)
        FROM mice
        INNER JOIN mouse_weights
        ON mice.mouse_id = mouse_weights.mouse_id
        WHERE mouse_name = %s
        """, args=[mouse_name], verbose=False)
    print(f'\naverage weight: {average.values()[0]}\n\n')


def insert_mouse_weight(mouse_name, weight, date=None):
    """ Insert a new weight record into the mouse weight database. Calls
    read_mouse_weights before completing.

    Parameters
    ----------
    mouse_name : str
        Name of the mouse corresponding to this weight. Matches mouse_name
        in the mice table.
    weight : float
        Weight value to add to the mouse_wieghts table in the database.
    date : str, optional
        Date to associate the weight entry with. Should be in the mySQL date
        format, YYYY-MM-DD HH:MM:SS. Defaults to now.
    """

    db = ExperimentDatabase()

    mouse_ID = int(db.select("""
        SELECT mouse_id
        FROM mice
        WHERE `mouse_name` = %s
        """, args=[mouse_name]).values()[0])

    if date is None:
        db.query("""
            INSERT INTO mouse_weights (`mouse_id`,`weight`)
            VALUES (%s, %s)
            """, [mouse_ID, weight], verbose=False)
    else:
        db.query("""
            INSERT INTO mouse_weights (`mouse_id`,`weight`,`date`)
            SELECT `mouse_id`, %s, %s
            FROM mice
            WHERE `mouse_name` = %s
            """, [weight, date, mouse_name], verbose=False)

    read_mouse_weights(mouse_name)


# def pair_imaging_data(imaging_path, trial_id=None, behavior_file=None,
#                       start_time=None):
#     """ Pair a tSeries directory with a trial record in the database.
#
#     Parameters
#     ----------
#     imaging_path : str
#         tSeries directory to add to trial record.
#     trial_id : int, optional
#         Trial id from the SQL trials table to match SQL record. Defaults to
#         None.
#     behavior_file : str, optional
#         Path to a behavior file which matches a SQL record. Defaults to None.
#     start_time : str, optional
#         Date time string to match a to  SQL record. Defaults to None.
#     """
#
#     db = ExperimentDatabase()
#
#     if trial_id and behavior_file:
#         raise ValueError('Cannot use both trial id and behavior file path')
#     elif trial_id is not None:
#         db.query("""
#             UPDATE trials
#             SET `tSeries_path` = %s
#             WHERE `trial_id` = %s
#             """, args=[imaging_path, trial_id], verbose=False)
#     elif behavior_file is not None:
#         behavior_file = os.path.splitext(behavior_file)[0] + '%'
#         db.query("""
#             UPDATE trials
#             SET `tSeries_path` = %s
#             WHERE behavior_file LIKE %s
#         """, args=[imaging_path, behavior_file], verbose=False)
#         trial_id = fetch_trial_id(behavior_file=behavior_file)
#
#     # TODO: should be start_time/mouse_name combination to ensure this value
#     # is unique.
#     elif start_time is not None:
#         start_time = _resolve_start_time(start_time)
#         # TODO: experiments table is not working (or used) necessary for
#         # legacy saliance compatibilty. Update or remove...
#
#         # expt_id = fetch_experiment_ID(start_time)
#         # if expt_id is not None:
#         #     db.query("""
#         #         UPDATE trials
#         #         SET `tSeries_path` = %s
#         #         WHERE `experiment_id` = %s
#         #     """, args=[imaging_path, expt_id], verbose=False)
#         # else:
#         db.query("""
#             UPDATE trials
#             SET `tSeries_path` = %s
#             WHERE `start_time` = %s
#         """, args=[imaging_path, start_time], verbose=False)
#
#     else:
#         raise ValueError('Need either trial id or behavior file path')
#
#     db.disconnect()


def update_behavior_filename(old_filename, new_filename):
    """ Update the behavior_file field of a SQL record in the trials database.

    Parameters
    ----------
    old_filename : str
        The file path currently stored in the SQL database to look up the
        record by.
    new_filename : str
        The new path to change the behavior_file record to.
    """

    db = ExperimentDatabase()
    db.query("""
        UPDATE trials
        SET `behavior_file` = %s
        WHERE behavior_file = %s
        """, args=[new_filename, old_filename])

    db.disconnect()


def update_trial(behavior_file, mouse_name, start_time, stop_time,
                 experiment_group, trial_id=None):
    """
    Parameters
    ----------
    behavior_file : str
        Path to behavior file.
    mouse_name : str
    start_time : str
    stop_time : str
    experiment_group : str
    trial_id : int, optional
        Target trial to update. If None, a new trial is created."""

    mouse_id = fetch_mouse_ID(mouse_name, create=True)
    if trial_id is None:
        trial_id = fetch_trial_id(behavior_file=behavior_file)
    start_time = _resolve_start_time(start_time)
    stop_time = _resolve_start_time(stop_time)

    db = ExperimentDatabase()
    # if this behavior_file path is alread associated with the trial, update
    # the existing record, otherwise create a new one.
    if trial_id is not None:
        db.query("""
            UPDATE trials
            SET `mouse_id` = %s,
                `start_time` = %s,
                `stop_time` = %s,
                `behavior_file` = %s,
                `experiment_group` = %s
            WHERE trial_id = %s
            """, args=[mouse_id, start_time, stop_time, behavior_file,
                 experiment_group, trial_id])
    else:
        db.query("""
            INSERT INTO trials
                (`mouse_id`, `start_time`, `stop_time`, `behavior_file`,
                 `experiment_group`)
            VALUES (%s, %s, %s, %s, %s)
            """, args=[mouse_id, start_time, stop_time, behavior_file,
                 experiment_group])

    db.disconnect()


def update_attr(table, group_name, group_id, attribute, value):
    """
    Parameters
    ----------
    table :
    group_name :
    group_id :
    attribute : str
        Attribute in database to set
    value : str
        Value to set for attribute
    """

    db = ExperimentDatabase()
    attr = db.select("""
        SELECT attribute_id
        FROM {}
        WHERE {} = %s
        AND attribute = %s
        """.format(table, group_name), args=[group_id, attribute],
        verbose=False)

    if attr is not None:
        db.query("""
            UPDATE {}
            SET value = %s
            WHERE attribute_id = %s
            """.format(table),
                 args=[value, attr['attribute_id']], verbose=False)
    else:
        db.query("""
            INSERT INTO {} (`{}`, `attribute`, `value`)
            VALUES (%s, %s, %s)
            """.format(table, group_name),
                 args=[group_id, attribute, value], verbose=False)

    db.disconnect()


def fetch_attr(table, group_name, group_id, attribute, default=None):
    """
    Parameters
    ----------
    table :
    group_name :
    group_id :
    attribute : str
        Attribute to fetch
    default : optional
        What to return if attribute does not exist.
    """
    db = ExperimentDatabase()
    attr = db.select("""
        SELECT value
        FROM {}
        WHERE {} = %s
        AND attribute = %s
        """.format(table, group_name),
        args=[group_id, attribute], verbose=False)
    db.disconnect()

    if attr:
        return attr['value']

    return default


def _delete_all_attrs(table, group_name, group_id):
    db = ExperimentDatabase()
    db.query("""
        DELETE FROM {}
        WHERE {} = %s
        """.format(table, group_name), args=[group_id], verbose=False)
    db.disconnect()


def _delete_attr(table, group_name, group_id, attribute):
    db = ExperimentDatabase()
    db.query("""
        DELETE FROM {}
        WHERE {} = %s
        AND attribute = %s
        """.format(table, group_name),
             args=[group_id, attribute], verbose=False)
    db.disconnect()


def update_mouse_attr(mouse_name, attribute, value):
    mouse_id = fetch_mouse_ID(mouse_name)
    update_attr('mouse_attributes', 'mouse_id', mouse_id, attribute, value)


def update_mouse_page(mouse_name, attribute, value):
    mouse_id = fetch_mouse_ID(mouse_name)
    if type(value) != str:
        value = json.dumps(value)

    update_attr('mouse_pages', 'mouse_id', mouse_id, attribute, value)


def fetch_mouse_attr(mouse_name, attribute, default=None):
    mouse_id = fetch_mouse_ID(mouse_name)
    return fetch_attr(
        'mouse_attributes', 'mouse_id', mouse_id, attribute, default=default)


def delete_mouse_attr(mouse_name, attribute, default=None):
    mouse_id = fetch_mouse_ID(mouse_name)
    _delete_attr(
        'mouse_attributes', 'mouse_id', mouse_id, attribute)


def fetch_mouse_page_attr(mouse_name, attribute, default=None):
    mouse_id = fetch_mouse_ID(mouse_name)
    attr = fetch_attr(
        'mouse_pages', 'mouse_id', mouse_id, attribute, default=default)
    if type(attr) == str:
        try:
            attr = json.loads(attr)
        except ValueError:
            pass

    return attr


def delete_mouse_page_attr(mouse_name, attribute):
    mouse_id = fetch_mouse_ID(mouse_name)
    _delete_attr(
        'mouse_pages', 'mouse_id', mouse_id, attribute)


def update_trial_attr(trial_id, attribute, value):
    update_attr('trial_attributes', 'trial_id', trial_id, attribute, value)


def fetch_trial_attr(trial_id, attribute, default=None):
    return fetch_attr(
        'trial_attributes', 'trial_id', trial_id, attribute, default=default)


def delete_trial_attr(trial_id, attribute):
    _delete_attr(
        'trial_attributes', 'trial_id', trial_id, attribute)


def update_virus_attr(trial_id, attribute, value):
    update_attr('virus_attributes', 'virus_id', trial_id, attribute, value)


def fetch_virus_attr(trial_id, attribute, default=None):
    return fetch_attr(
        'virus_attributes', 'virus_id', trial_id, attribute, default=default)


def delete_virus_attr(trial_id, attribute):
    _delete_attr(
        'virus_attributes', 'virus_id', trial_id, attribute)


def _fetch_all_attrs(table, id_field, id_value, parse=False):
    db = ExperimentDatabase()
    attrs = db.select_all("""
        SELECT attribute, value
        FROM {}
        WHERE {} = %s
        """.format(table, id_field), args=[id_value], verbose=False)
    db.disconnect()

    attrs = {entry['attribute']: entry['value'] for entry in attrs}
    if parse:
        for key in attrs.keys():
            try:
                attrs[key] = json.loads(attrs[key])
            except ValueError:
                pass

    return attrs


def fetch_all_virus_attrs(virus_id, parse=True):
    attrs = _fetch_all_attrs('virus_attributes', 'virus_id', virus_id)

    return attrs


def fetch_all_mouse_attrs(mouse_name, parse=False):
    if isinstance(mouse_name, int):
        mouse_id = mouse_name
    else:
        mouse_id = fetch_mouse_ID(mouse_name)

    return _fetch_all_attrs('mouse_attributes', 'mouse_id', mouse_id,
                            parse=parse)


def fetch_all_trial_attrs(trial_id, parse=False):
    db = ExperimentDatabase()
    attrs = db.select_all("""
        SELECT attribute, value
        FROM trial_attributes
        WHERE trial_id = %s
        """, args=[trial_id], verbose=False)
    db.disconnect()

    if parse:
        for i, entry in enumerate(attrs):
            try:
                attrs[i]['value'] = json.loads(entry['value'])
            except ValueError:
                pass

    return {entry['attribute']: entry['value'] for entry in attrs}


def delete_all_trial_attrs(trial_id):
    _delete_all_attrs('trial_attributes', 'trial_id', trial_id)


def delete_all_mouse_attrs(mouse_id):
    _delete_all_attrs('mouse_attributes', 'mouse_id', mouse_id)
    _delete_all_attrs('mouse_pages', 'mouse_id', mouse_id)


def _project_filter_sql(project_name):
    return """
        (SELECT DISTINCT m.*
         FROM mice m
         LEFT JOIN trials t
            ON m.mouse_id=t.mouse_id
         WHERE experiment_group='{0}'
         UNION
         SELECT DISTINCT m.*
         FROM mice m
         LEFT JOIN mouse_attributes ma
            ON m.mouse_id=ma.mouse_id
         WHERE (attribute='project_name' AND value='{0}')
         )
    """.format(project_name)


def delete_trial(trial_id):
    delete_all_trial_attrs(trial_id)

    db = ExperimentDatabase()
    db.query("""
        DELETE FROM trials
        WHERE trial_id = %s
        """, args=[trial_id])


def delete_mouse(mouse_id):
    delete_all_mouse_attrs(mouse_id)

    db = ExperimentDatabase()
    db.query("""
        DELETE FROM mice
        WHERE mouse_id = %s
        """, args=[mouse_id], verbose=False)


def fetch_mouse_ID(mouse_name, create=False, project_name=None):
    db = ExperimentDatabase()

    if project_name is not None:
        mouse_id = db.select_all("""
            SELECT DISTINCT mice.mouse_id
            FROM mice
            INNER JOIN trials
                ON trials.mouse_id=mice.mouse_id
            WHERE mouse_name = %s
                AND experiment_group = %s
            UNION
            SELECT mice.mouse_id
            FROM mice
            INNER JOIN mouse_attributes
                ON mice.mouse_id = mouse_attributes.mouse_id
            WHERE mouse_name = %s
                AND attribute = %s
                AND value = %s
            """, args=[mouse_name, project_name, mouse_name, 'project_name',
                       project_name])

    else:
        mouse_id = db.select_all("""
            SELECT mouse_id
            FROM mice
            WHERE mouse_name = %s
            """, args=[mouse_name])

    if create and len(mouse_id) == 0:
        db.query("""
            INSERT INTO mice (mouse_name)
            VALUES (%s)
        """, args=[mouse_name])
        db.disconnect()
        if project_name is not None:
            update_mouse_attr(mouse_name, 'project_name', project_name)
        mouse_id = fetch_mouse_ID(mouse_name, create=False)
    elif len(mouse_id) == 1:
        db.disconnect()
        mouse_id = int(list(mouse_id[0].values())[0])
    else:
        raise KeyError('unable to uniquely identify mouse {}'.format(
            mouse_name))

    return mouse_id


def fetch_mouse(mouse_id):
    db = ExperimentDatabase()
    mouse = db.select("""
        SELECT *
        FROM mice
        WHERE mouse_id = %s
    """, args=[mouse_id])

    return mouse


def fetch_trial(trial_id):
    db = ExperimentDatabase()
    trial = db.select("""
        SELECT *
        FROM trials
        WHERE trial_id = %s
        """, args=[trial_id])
    return trial


def fetch_trial_id(behavior_file=None, tSeries_path=None, mouse_name=None,
                   startTime=None):
    db = ExperimentDatabase()

    arg_names = ['tSeries_path', 'behavior_file', 'mouse_name', 'startTime']
    arg_vals = [tSeries_path, behavior_file, mouse_name, startTime]
    arg = [n for n, v in zip(arg_names, arg_vals) if v is not None]

    if 'startTime' in arg:
        trial_id = db.select("""
            SELECT trial_id FROM trials
            INNER JOIN mice
                ON trials.mouse_id=mice.mouse_id
            WHERE mouse_name = %s
                AND start_time = %s
        """, args=[mouse_name, _resolve_start_time(startTime)])
    elif 'startTime' in arg:
        raise Exception("require mouse_id")
    else:
        trial_id = db.select("""
            SELECT trial_id FROM trials
            WHERE {} = %s
            """.format(arg[0]), args=[eval(arg[0])])

    if trial_id is None:
        return None

    return int(trial_id['trial_id'])


def fetch_imaged_trials(mouse_name):
    db = ExperimentDatabase()
    trials = db.select_all("""
        SELECT trial_id, start_time, mouse_name, behavior_file, tSeries_path
        FROM trials
        INNER JOIN mice
        ON mice.mouse_id = trials.mouse_id
        WHERE tSeries_path IS NOT NULL
        AND mouse_name = %s
        ORDER BY start_time ASC
        """, args=[mouse_name])
    db.disconnect()

    return trials


def fetch_attribute_values(attr, project_name=None):
    db = ExperimentDatabase()

    table = 'mice'
    if attr in db._trial_fields:
        condition = ''
        if project_name is not None:
            condition = 'WHERE experiment_group=\'{}\''.format(project_name)

        values = db.select_all("""
            SELECT DISTINCT {} FROM trials t
            LEFT JOIN mice m
                ON t.mouse_id=m.mouse_id
            {}
        """.format(attr, condition))
        return [value[attr] for value in values]

    if project_name is not None:
        table = _project_filter_sql(project_name)

    values = db.select_all("""
        SELECT DISTINCT value FROM {0} AS m
        LEFT JOIN trials t
            ON m.mouse_id=t.mouse_id
        LEFT JOIN trial_attributes ta
            ON t.trial_id=ta.trial_id
        WHERE attribute=%s
        UNION
        SELECT DISTINCT value FROM {0} AS m
        LEFT JOIN mouse_attributes ma
            ON m.mouse_id=ma.mouse_id
        WHERE attribute=%s
    """.format(table), args=[attr, attr])

    return [value['value'] for value in values]


def fetch_trials_with_attr_value(attr, value):
    db = ExperimentDatabase()
    trials = db.select_all("""
        SELECT trial_id
        FROM trial_attributes
        WHERE attribute = %s AND value = %s
        ORDER BY trial_id ASC
        """, args=[attr, value])
    db.disconnect()

    return [int(trial['trial_id']) for trial in trials]


def fetch_mice(*args, **kwargs):
    db = ExperimentDatabase()
    query_string = """
        SELECT DISTINCT m.mouse_id
        FROM {0} AS m
        LEFT JOIN mouse_attributes ma
            ON m.mouse_id=ma.mouse_id
        LEFT JOIN trials t
            ON m.mouse_id=t.mouse_id
        WHERE {1}
        UNION
        SELECT DISTINCT m.mouse_id
        FROM {0} AS m
        LEFT JOIN trials t
            ON m.mouse_id=t.mouse_id
        LEFT JOIN trial_attributes ta
            ON t.trial_id=ta.trial_id
        WHERE {1}
        """

    table = 'mice'
    if 'project_name' in kwargs.keys():
        table = _project_filter_sql(kwargs['project_name'])
        del kwargs['project_name']

    if 'experiment_group' in kwargs.keys():
        table = _project_filter_sql(kwargs['experiment_group'])
        del kwargs['experiment_group']

    trial_conditions = []
    conditions = []
    query_args = []
    trial_condition = "({} IS NOT NULL)"
    condition_string = "(attribute='{}')"
    for key in args:
        if key in db._trial_fields:
            if key == 'mouse_id':
                key = 'm.%s' % key

            trial_conditions.append(trial_condition.format(key))
        else:
            conditions.append(condition_string.format(key))

    trial_condition = "({}={})"
    condition_string = "(attribute='{}' AND value={})"
    for key, values in kwargs.items():
        if type(values) != list:
            values = [values]

        alternatives = []
        trial_cond = False
        for val in values:
            if key in ['start_time', 'stop_time']:
                val = _resolve_start_time(val)

            try:
                float(val)
            except BaseException:
                val = "'{}'".format(val)

            if key in db._trial_fields:
                trial_cond = True
                if key == 'mouse_id' or key == 'trial_id':
                    _key = 't.%s' % key
                else:
                    _key = key

                if _key == 'tSeries_path':
                    alternatives.append('tSeries_path LIKE %s')
                    query_args.append(os.path.normpath(val[1:-1] + '%'))
                else:
                    alternatives.append(trial_condition.format(_key, val))
            else:
                alternatives.append(condition_string.format(key, val))
        if trial_cond:
            trial_conditions.append("({})".format(" OR ".join(alternatives)))
            trial_cond = False
        else:
            conditions.append("({})".format(" OR ".join(alternatives)))

    query = None
    if len(trial_conditions):
        query = """
            SELECT DISTINCT m.*
            FROM {} m
            INNER JOIN trials t
                ON m.mouse_id=t.mouse_id
            WHERE {}
        """.format(table, " AND ".join(trial_conditions))

    for condition in conditions:
        if query is None:
            query = query_string.format(table, condition)
        else:
            query = """
                SELECT DISTINCT m.*
                FROM ({}) AS m
                INNER JOIN ({}) as n
                ON m.mouse_id=n.mouse_id
                """.format(query, query_string.format('mice', condition))

    if query is None:
        query = """
            SELECT DISTINCT m.*
            FROM {} AS m
        """.format(table)

    records = db.select_all(query, args=query_args)
    return [int(record['mouse_id']) for record in records]


def fetch_trials(*args, **kwargs):
    db = ExperimentDatabase()

    query_string = """
        SELECT DISTINCT t.*
        FROM {}
        LEFT JOIN trial_attributes ta
            ON t.trial_id = ta.trial_id
        LEFT JOIN mice m
            ON t.mouse_id = m.mouse_id
        WHERE {} ORDER BY start_time"""

    conditions = []
    query_args = []
    trial_condition = "({} IS NOT NULL)"
    condition_string = "(attribute='{}')"
    for key in args:
        if key in db._trial_fields:
            if key == 'mouse_id':
                key = 'trials.%s' % key
            conditions.append(trial_condition.format(key))
        else:
            conditions.append(condition_string.format(key))

    trial_condition = "({}={})"
    condition_string = "(attribute='{}' AND value={})"
    for key, values in kwargs.items():
        if type(values) != list:
            values = [values]

        alternatives = []
        for val in values:
            if key in ['start_time', 'stop_time']:
                val = _resolve_start_time(val)

            try:
                float(val)
            except BaseException:
                val = "'{}'".format(val)

            if key in db._trial_fields:
                if key == 'mouse_id' or key == 'trial_id':
                    _key = 't.%s' % key
                else:
                    _key = key

                if _key == 'tSeries_path':
                    if val[1:-1].strip() == '':
                        alternatives.append('tSeries_path IS NOT NULL')
                    else:
                        alternatives.append('tSeries_path LIKE %s')
                        query_args.append(os.path.normpath(val[1:-1]) + '%')
                else:
                    alternatives.append(trial_condition.format(_key, val))
            else:
                alternatives.append(condition_string.format(key, val))
        conditions.append("({})".format(" OR ".join(alternatives)))

    query = 'trials t'
    for condition in conditions[::-1]:
        query = " (" + query_string.format(query, condition) + ") AS t"
    query = query.rstrip(" AS t")

    trials = db.select_all(query, args=query_args)
    return [int(trial['trial_id']) for trial in trials]


def fetch_mouse_trials(mouse_name):
    db = ExperimentDatabase()
    trials = db.select_all("""
        SELECT trial_id, start_time, mouse_name, behavior_file, tSeries_path
        FROM trials
        INNER JOIN mice
        ON mice.mouse_id = trials.mouse_id
        AND mouse_name = %s
        ORDER BY start_time ASC
        """, args=[mouse_name])
    db.disconnect()

    return trials


def _resolve_start_time(start_time):
    try:
        tstruct = time.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass
    else:
        return start_time

    formats = ['%Y-%m-%d-%Hh%Mm%Ss', '%m/%d/%Y']
    for t_format in formats:
        try:
            tstruct = time.strptime(start_time, t_format)
            start_time = time.strftime('%Y-%m-%d %H:%M:%S', tstruct)
        except BaseException:
            pass
        else:
            return start_time

    raise Exception('unable to parse time')


def fetch_experiment_ID(start_time):
    db = ExperimentDatabase()
    start_time = _resolve_start_time(start_time)

    experiment_id = db.select("""
        SELECT experiment_id
        FROM experiments
        WHERE start_time = %s
        """, args=[start_time])

    if experiment_id is None:
        experiment_id = db.select("""
            SELECT trial_id
            FROM trials
            WHERE start_time = %s
            """, args=[start_time])

        if experiment_id is None:
            return None

        return int(experiment_id.values()[0])

    return int(experiment_id.values()[0])


def fetch_all_projects():
    db = ExperimentDatabase()
    projects = [record['experiment_group'] for record in
                db.select_all("""
                    SELECT DISTINCT experiment_group
                    FROM trials
                    ORDER BY experiment_group ASC
                """)]
    db.disconnect()

    return projects


def fetch_all_mice(project_name=None):
    db = ExperimentDatabase()

    if project_name is not None:
        mice = [record['mouse_name'] for record in
                db.select_all("""
                    SELECT DISTINCT m.mouse_name
                    FROM mice m
                    LEFT JOIN trials t
                       ON m.mouse_id=t.mouse_id
                    WHERE experiment_group='{0}'
                    UNION
                    SELECT DISTINCT m.mouse_name
                    FROM mice m
                    LEFT JOIN mouse_attributes ma
                       ON m.mouse_id=ma.mouse_id
                    WHERE (attribute='project_name' AND value='{0}')
                    ORDER BY mouse_name
                    """.format(project_name))]
    else:
        mice = [record['mouse_name'] for record in
                db.select_all("""
                    SELECT DISTINCT mouse_name
                    FROM mice
                    ORDER BY mouse_name ASC
                """)]

    return mice


def insert_into_experiment(trial_id, start_time, stop_time=None):
    db = ExperimentDatabase()
    start_time = _resolve_start_time(start_time)

    experiment_id = fetch_experiment_ID(start_time)
    if experiment_id is None:
        db.query("""
            INSERT INTO experiments (start_time)
            VALUES (%s)
            """, args=[start_time])

        experiment_id = fetch_experiment_ID(start_time)

    db.query("""
        UPDATE trials
        SET experiment_id = %s
        WHERE trial_id = %s
        """, args=[experiment_id, trial_id])

    if stop_time is not None:
        db.query("""
            UPDATE experiments
            SET stop_time = %s
            WHERE experiment_id = %s
            """, args=[experiment_id, trial_id])

    db.disconnect()


def fetchVirus(virus_id):
    db = ExperimentDatabase()
    virus = db.select("""
        SELECT *
        FROM viruses
        WHERE virus_id = %s
        """, args=[virus_id])
    db.disconnect()
    return virus


def update_virus(virus_id, name=None, arrival_date=None, source_code=None):
    db = ExperimentDatabase()
    if arrival_date is not None:
        arrival_date = _resolve_start_time(arrival_date)

    id_check = db.select("""
        SELECT COUNT(virus_id)
        FROM viruses
        WHERE virus_id = %s
        """, args=[virus_id])

    if id_check.values()[0] == 0:
        db.query("""
            INSERT INTO viruses
            VALUES (%s, %s, %s, %s)
            """, args=[virus_id, name, arrival_date, source_code])
    else:
        args = ['name', 'arrival_date', 'source_code']
        arg = ['{} = %s'.format(a) for a in args if eval(a) is not None]
        vals = [eval(a) for a in args if eval(a) is not None]
        db.query("""
            UPDATE viruses
            SET {}
            WHERE virus_id = %s
            """.format(', '.join(arg)), args=(vals + [virus_id]))

    db.disconnect()
