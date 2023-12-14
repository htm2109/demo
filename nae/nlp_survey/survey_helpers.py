

oe_dict = {
    'Teaching quality or Staff': ['teacher', 'teachers', 'education quality', 'homework', 'learning', 'teaching quality', 'staff'],
    'School facilities': ['cafeteria', 'lunch', 'food', 'building'],
    'Communication or Parental involvement': ['communication', 'parental involvement', 'parents', 'community'],
    'Educational philosophy': ['culture', 'philosophy', 'Britishness', 'pedagogy', 'curriculum'],
    'Cost': ['cost', 'tuition', 'fees', 'price']
}

columns_to_keep = ['school', 'phase', 'category', 'label']

label_columns = [
    'label_nps_follow_up_translated',
    'label_value_most_about_school_translated',
    'label_where_school_needs_to_develop_translated',
    'label_multiple_children_feedback_translated'
]

category_columns = [
    'Primary_OE_nps_follow_up_translated', 'Secondary_OE_nps_follow_up_translated',
    'Primary_OE_value_most_about_school_translated', 'Secondary_OE_value_most_about_school_translated',
    'Primary_OE_where_school_needs_to_develop_translated', 'Secondary_OE_where_school_needs_to_develop_translated',
    'Primary_OE_multiple_children_feedback_translated', 'Secondary_OE_multiple_children_feedback_translated'
]

translated_responses = ['nps_follow_up_translated', 'value_most_about_school_translated',
                        'where_school_needs_to_develop_translated', 'multiple_children_feedback_translated']
