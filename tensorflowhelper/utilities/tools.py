from .Error import TFHError

DEBUG = True

def validate(where, describe_elem, expect_elem, got_elem):
    if( DEBUG ):
        try:
            result = _validate(where, describe_elem, expect_elem, got_elem)
            print("PASSED Validation in {} of {} of {} and {}".format(where, describe_elem, expect_elem, got_elem))
            return result
        except TFHError:
            print("FAILED Validation in {} of {} of {} and {}".format(where, describe_elem, expect_elem, got_elem))
            pass
            
        raise TFHError(
                where,
                "{} is not compatible".format(describe_elem),
                "Expect {} : {}".format(describe_elem, repr(expect_elem)),
                "Got {} : {}".format(describe_elem, repr(got_elem)) )
    
    else:
        return _validate(where, describe_elem, expect_elem, got_elem)

def _validate(where, describe_elem, expect_elem, got_elem):
    if( expect_elem != None and 
        # got_elem != None and 
        (type(expect_elem) is type(got_elem) and expect_elem != got_elem) ):
        if( isinstance(expect_elem,(list)) and isinstance(got_elem,(list)) and len(expect_elem) == len(got_elem)):
            try:
                for exp, got in zip(expect_elem, got_elem):
                    _validate(where, describe_elem, exp, got)
                return
            except TFHError:
                pass
                
        raise TFHError(
                where,
                "{} is not compatible".format(describe_elem),
                "Expect {} : {}".format(describe_elem, repr(expect_elem)),
                "Got {} : {}".format(describe_elem, repr(got_elem)) )