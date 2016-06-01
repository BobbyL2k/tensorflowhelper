from .Error import TFHError

DEBUG = True

def validate(where, describe_elem, expect_elem, got_elem):
    if DEBUG:
        try:
            _validate(where, describe_elem, expect_elem, got_elem)
            print("PASSED Validation in {} of {} of {} and {}".format(
                where, describe_elem, expect_elem, got_elem))

        except TFHError as tfh_error:
            print("FAILED Validation in {} of {} of {} and {}".format(
                where, describe_elem, expect_elem, got_elem))
            raise tfh_error

    else:
        _validate(where, describe_elem, expect_elem, got_elem)

def _validate(where, describe_elem, expect_elem, got_elem):
    pass_validation = True

    if expect_elem is not None:
        if isinstance(expect_elem, (list, tuple)) and isinstance(got_elem, (list, tuple)):
            if len(expect_elem) == len(got_elem):
                try:
                    for exp, got in zip(expect_elem, got_elem):
                        _validate(where, describe_elem, exp, got)
                except TFHError:
                    pass_validation = False
            else: pass_validation = False

        elif type(expect_elem) is type(got_elem) and expect_elem != got_elem:
            pass_validation = False

    if not pass_validation:
        raise TFHError(
            where,
            "{} is not compatible".format(describe_elem),
            "Expect {} : {}".format(describe_elem, repr(expect_elem)),
            "Got {} : {}".format(describe_elem, repr(got_elem)))

def validate_tf_input(name, tf_layer_input, shape, dtype):
    """Validate if the TensorFlow variable is compatible with
    given shape and data type"""
    validate(name, "Tensor shape", shape, tf_layer_input._shape_as_list())
    validate(name, "Tensor dtype", dtype, tf_layer_input.dtype)
