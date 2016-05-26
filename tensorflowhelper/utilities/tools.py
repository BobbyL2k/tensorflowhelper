from .Error import TFHError

def validatie(where, describe_elem, expect_elem, got_elem):
    if( expect_elem != None and got_elem != None and expect_elem != got_elem ):
        if( isinstance(expect_elem,list) and isinstance(expect_elem,list) and len(expect_elem) == len(got_elem)):
            try:
                for exp, got in zip(expect_elem, got_elem):
                     validatie(where, describe_elem, exp, got)
                return
            except TFHError:
                pass
                
        raise TFHError(
                where,
                "{} is not compatible".format(describe_elem),
                "Expect {} : {}".format(describe_elem, repr(expect_elem)),
                "Got {} : {}".format(describe_elem, repr(got_elem)) )