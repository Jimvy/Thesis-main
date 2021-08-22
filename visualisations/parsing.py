import argparse

from utils_vizigoth import error

ALL_POSSIBLE_VARS = ('as_norm', 'as_angle', 'as_x', 'as_y', 'at_norm', 'at_angle', 'at_x', 'at_y', 'ps_x', 'ps_y', 'pt_x', 'pt_y', 'temp')
DISPLAYNAMES = {
    'as_x': r'$a_S[1]$',
    'as_y': r'$a_S[2]$',
    'as_norm': r'$|a_S|$',
    'as_angle': r'$\angle a_S$',
    'ps_x': r'$p_S[1]$',
    'ps_y': r'$p_S[2]$',
    'at_x': r'$a_T[1]$',
    'at_y': r'$a_T[2]$',
    'at_norm': r'$|a_T|$',
    'at_angle': r'$\angle a_T$',
    'pt_x': r'$p_T[1]$',
    'pt_y': r'$p_T[2]$',
    'temp': r'$\tau$'
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plots the loss in a binary classification task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--temp', type=float, help='Value for the temperature')
    parser.add_argument('--as-norm', type=float,
                        help='Value for the norm of the student activation (polar coords)')
    parser.add_argument('--as-angle', type=float,
                        help='Value for the angle (in degrees) of the student activation (polar coords)')
    parser.add_argument('--as-x', '--as-0', type=float,
                        help='Value for the first coord of the student activation [0]')
    parser.add_argument('--as-y', '--as-1', type=float,
                        help='Value for the second coord of the student activation [1]')
    parser.add_argument('--at-norm', type=float,
                        help='Value for the norm of the teacher activation (polar coords)')
    parser.add_argument('--at-angle', type=float,
                        help='Value for the angle (in degrees) of the teacher activation (polar coords)')
    parser.add_argument('--at-x', '--at-0', type=float,
                        help='Value for the first coord of the teacher activation [0]')
    parser.add_argument('--at-y', '--at-1', type=float,
                        help='Value for the second coord of the teacher activation [1]')
    parser.add_argument('--ps-x', '--ps-0', type=float,
                        help='Value for the probability of the first class for the student')
    parser.add_argument('--ps-y', '--ps-1', type=float,
                        help='Value for the probability of the second class for the student')
    parser.add_argument('--pt-x', '--pt-0', type=float,
                        help='Value for the probability of the first class for the teacher')
    parser.add_argument('--pt-y', '--pt-1', type=float,
                        help='Value for the probability of the second class for the teacher')
    parser.add_argument('--loss', dest='losses_list', action='append', nargs='+',
                        help='Which loss(es) to plot.'
                             'Default is CE, if you want to plot more you need to specify its name and its weight.'
                             'You also need to re-specify CE if you still want it.'
                             'There are additional options for some losses (ex HKD for the power of the tau factor')
    parser.add_argument('--separate', action='store_true',
                        help='If the different losses should also be plotted on separate graphs')
    parser.add_argument('--vary', dest='vary_vars_list', nargs='+', action='append', required=True,
                        help='Which variable to vary in the graph. Can be repeated at most two times.'
                             '1st arg is name of variable, 2nd arg is start val, 3rd arg is end val.'
                             '4th arg is optional and defaults to num, meaning that the 5th arg is the number of samples.'
                             'If given, then the 5th arg must be given too (default is 50).'
                             'Other valid values are \'step\', in which case the 5th arg must be the step between samples.')
    args = parser.parse_args()
    # Let's check the validity of the arguments

    if len(args.vary_vars_list) > 2:
        error("Error: you can't specify more than 2 variables to vary")
    if len(args.vary_vars_list) <= 0:
        error("Error: you must specify at least one variable to vary")
    # Parse the vary list and check it
    for idx, vary_var_l in enumerate(args.vary_vars_list):
        vary_var_args = {}
        if len(vary_var_l) < 3:
            error(f"Error: you must specify at least 3 parameters for the {idx+1}-th variable")
        var_name = vary_var_l[0]
        if var_name[2:] == '-0':
            var_name = var_name[:2] + '-x'
        if var_name[2:] == '-1':
            var_name = var_name[:2] + '-y'
        vary_var_args['name'] = var_name.replace('-', '_')
        if vary_var_args['name'] not in ALL_POSSIBLE_VARS:
            error(f"Error: the variable you asked to vary \'{vary_var_l[0]}\' is not available")
        if getattr(args, vary_var_args['name']) is not None:
            error(f"Error: you specified \'{var_name}\' as a variable, yet you gave a fixed value for it afterwards")
        if vary_var_l[1] == 'log':
            vary_var_args['scale'] = 'log'
            del vary_var_l[1]
        else:
            vary_var_args['scale'] = 'linear'
            try:
                float(vary_var_l[1])
            except ValueError:
                print(f"Warning: don't know about option \'{vary_var_l[1]}\', you can either put a float or 'log'")
                del vary_var_l[1]  # Don't care other values
        vary_var_args['start'] = float(vary_var_l[1])
        vary_var_args['end'] = float(vary_var_l[2])
        if len(vary_var_l) >= 4:
            if len(vary_var_l) < 5:
                error(f"Error: in vary option, if you specify a 4th arg, then you must specify a 5th arg too")
            typ = vary_var_l[3]
            if typ not in ['num', 'step']:
                error("Error: the 4th arg must be \'num\' or \'step\'")
            vary_var_args['type'] = typ
            vary_var_args[typ] = float(vary_var_l[4])
        else:
            vary_var_args['type'] = 'num'
            vary_var_args['num'] = 50
        args.vary_vars_list[idx] = vary_var_args

    losses_to_display = []
    if not args.losses_list:
        args.losses_list = [['ce', '1']]
    needs_teacher = False
    for idx, loss_l in enumerate(args.losses_list):
        if len(loss_l) == 1:
            # only the loss name: must be CE
            loss_name = loss_l[0]
            if loss_name not in ['CE', 'ce']:
                error(f"Error: you must specify the weight for loss \'{loss_name}\'")
            losses_to_display.append(('CE', 1))
        elif len(loss_l) >= 2:
            loss_name = loss_l[0]
            loss_weight = float(loss_l[1])
            if len(loss_l) >= 3:
                more_ops = loss_l[2:]
            else:
                more_ops = []
            if loss_name in ['hkd', 'HKD']:
                needs_teacher = True
            if loss_name not in ['CE', 'ce', 'HKD', 'hkd']:
                error(f"Error: unknown loss name \'{loss_name}\'")
            losses_to_display.append((loss_name.upper(), loss_weight, more_ops))
        else:
            error(f"Error: unknown additional arguments for the loss")
    if len(losses_to_display) == 0:
        losses_to_display.append(('CE', 1))
    args.losses_list = losses_to_display

    for model_name in ['student', 'teacher']:
        model_name_f = model_name[0]
        acts_name = [f'a{model_name_f}_{x}' for x in ['norm', 'angle', 'x', 'y']]
        vars_acts_used = []
        for arg_name in acts_name:
            if getattr(args, arg_name) is not None:
                vars_acts_used.append(arg_name[3:])
            for arg in args.vary_vars_list:
                if arg['name'] == arg_name:
                    vars_acts_used.append(arg_name[3:])
        probs_names = [f'p{model_name_f}_{x}' for x in ['x', 'y']]
        probs_name_used = []
        for arg_name in probs_names:
            if getattr(args, arg_name) is not None:
                probs_name_used.append(arg_name[3:])
            for arg in args.vary_vars_list:
                if arg['name'] == arg_name:
                    probs_name_used.append(arg_name[3:])
        if len(vars_acts_used) not in (0, 2):
            error(f"Error: variable \'a{model_name_f}\' doesn't have exactly two components specified: it has {vars_acts_used} components.")
        if len(vars_acts_used) == 2:
            setattr(args, f'{model_name}_input_type', 'acts')
            # Then, that means we must be using student/teacher activations; prevent the use of probas
            if len(probs_name_used) > 0:
                error(f"Error: you shouldn't mix probas and activations for the {model_name}")
        elif len(vars_acts_used) == 0:
            setattr(args, f'{model_name}_input_type', 'probs')
            # Then, that means we must be using student/teacher probas; prevent the use of both probas
            if len(probs_name_used) != 1:
                if len(probs_name_used) > 1:
                    error(f"Error: you shouldn't put both probabilities for the {model_name}")
                elif not (model_name_f == 't' and not needs_teacher):
                    error(f"Error: you should provide some parameters for the {model_name}")
    # if args.student_input_type != args.teacher_input_type:
    #     error(f"For now we don't support mixing two input types together")

    uses_temp_1 = args.temp is not None
    uses_temp_2 = any([x['name'] == 'temp' for x in args.vary_vars_list])
    if uses_temp_1 and uses_temp_2:
        error("Error: temperature two times specified (and a logic error has been discovered")
    if not uses_temp_1 and not uses_temp_2:
        error("Error: temperature not specified")

    return args
