# import pandas as pd
# from pandas_schema import Column, Schema
#
# from pandas_schema.validation import LeadingWhitespaceValidation, TrailingWhitespaceValidation, CanConvertValidation, MatchesPatternValidation, InRangeValidation, InListValidation
#
# schema = Schema([
#     Column('Given Name', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
#     Column('Family Name', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
#     Column('Age', [InRangeValidation(0, 120)]),
#     Column('Sex', [InListValidation(['Male', 'Female', 'Other'])]),
#     Column('Customer ID', [MatchesPatternValidation(r'\d{4}[A-Z]{4}')])
# ])
#
# errors = schema.validate(test_data)
#
# for error in errors:
#     print(error)
#
# {row: 0, column: "Given Name"}: "Gerald " contains trailing whitespace
# {row: 1, column: "Age"}: "270" was not in the range [0, 120)
# {row: 1, column: "Sex"}: "male" is not in the list of legal options (Male, Female, Other)
# {row: 2, column: "Family Name"}: "Majewska " contains trailing whitespace
# {row: 2, column: "Customer ID"}: "775ANSID" does not match the pattern "\d{4}[A-Z]{4}"
#
#
