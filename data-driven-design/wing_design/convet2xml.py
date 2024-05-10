def convert2xml(path_prefix,
                sec1_,
                sec2_,
                sec3_,
                sec4_,
                sec5_):
    s = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE explane>
<explane version="1.0">
    <Units>
        <length_unit_to_meter>1</length_unit_to_meter>
        <mass_unit_to_kg>1</mass_unit_to_kg>
    </Units>
    <Plane>
        <Name>%s</Name>
        <Description></Description>
        <Inertia/>
        <has_body>false</has_body>
        <wing>
            <Name>Main Wing</Name>
            <Type>MAINWING</Type>
            <Color>
                <red>140</red>
                <green>152</green>
                <blue>168</blue>
                <alpha>255</alpha>
            </Color>
            <Description></Description>
            <Position>          0,           0,           0</Position>
            <Tilt_angle>  0.000</Tilt_angle>
            <Symetric>true</Symetric>
            <isFin>false</isFin>
            <isDoubleFin>false</isDoubleFin>
            <isSymFin>false</isSymFin>
            <Inertia>
                <Volume_Mass>  0.000</Volume_Mass>
            </Inertia>
            <Sections>
                <Section>
                    <y_position>  %.4f</y_position>
                    <Chord>  %.4f</Chord>
                    <xOffset>  %.4f</xOffset>
                    <Dihedral>  %.4f</Dihedral>
                    <Twist>  %.4f</Twist>
                    <x_number_of_panels>13</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>13</y_number_of_panels>
                    <y_panel_distribution>INVERSE SINE</y_panel_distribution>
                    <Left_Side_FoilName>D:/project/wings_design/%s_root.dat</Left_Side_FoilName>
                    <Right_Side_FoilName>D:/project/wings_design/%s_root.dat</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>  %.4f</y_position>
                    <Chord>  %.4f</Chord>
                    <xOffset>  %.4f</xOffset>
                    <Dihedral>  %.4f</Dihedral>
                    <Twist>  %.4f</Twist>
                    <x_number_of_panels>13</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>13</y_number_of_panels>
                    <y_panel_distribution>INVERSE SINE</y_panel_distribution>
                    <Left_Side_FoilName>D:/project/wings_design/%s_span30.dat</Left_Side_FoilName>
                    <Right_Side_FoilName>D:/project/wings_design/%s_span30.dat</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position> %.4f</y_position>
                    <Chord>  %.4f</Chord>
                    <xOffset>  %.4f</xOffset>
                    <Dihedral>  %.4f</Dihedral>
                    <Twist> %.4f</Twist>
                    <x_number_of_panels>13</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>13</y_number_of_panels>
                    <y_panel_distribution>INVERSE SINE</y_panel_distribution>
                    <Left_Side_FoilName>D:/project/wings_design/%s_span70.dat</Left_Side_FoilName>
                    <Right_Side_FoilName>D:/project/wings_design/%s_span70.dat</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position> %.4f</y_position>
                    <Chord>  %.4f</Chord>
                    <xOffset> %.4f</xOffset>
                    <Dihedral> %.4f</Dihedral>
                    <Twist> %.4f</Twist>
                    <x_number_of_panels>13</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>13</y_number_of_panels>
                    <y_panel_distribution>INVERSE SINE</y_panel_distribution>
                    <Left_Side_FoilName>D:/project/wings_design/%s_tip.dat</Left_Side_FoilName>
                    <Right_Side_FoilName>D:/project/wings_design/%s_tip.dat</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position> %.4f</y_position>
                    <Chord>  %.4f</Chord>
                    <xOffset> %.4f</xOffset>
                    <Dihedral> %.4f</Dihedral>
                    <Twist> %.4f</Twist>
                    <x_number_of_panels>13</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>13</y_number_of_panels>
                    <y_panel_distribution>INVERSE SINE</y_panel_distribution>
                    <Left_Side_FoilName>D:/project/wings_design/%s_winglet.dat</Left_Side_FoilName>
                    <Right_Side_FoilName>D:/project/wings_design/%s_winglet.dat</Right_Side_FoilName>
                </Section>
            </Sections>
        </wing>
    </Plane>
</explane>
    ''' % (path_prefix.split('/')[-1],
           sec1_[0], sec1_[1], sec1_[2], sec1_[3], sec1_[4], path_prefix, path_prefix,
           sec2_[0], sec2_[1], sec2_[2], sec2_[3], sec2_[4], path_prefix, path_prefix,
           sec3_[0], sec3_[1], sec3_[2], sec3_[3], sec3_[4], path_prefix, path_prefix,
           sec4_[0], sec4_[1], sec4_[2], sec4_[3], sec4_[4], path_prefix, path_prefix,
           sec5_[0], sec5_[1], sec5_[2], sec5_[3], sec5_[4], path_prefix, path_prefix,
           )
    return s
