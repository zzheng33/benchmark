variable ibead uloop 99 pad

units metal
atom_style atomic
atom_modify map yes
boundary p p p
pair_style lj/cut 9.5251
read_data data.metalnpt${ibead}

pair_coeff    * * 0.00965188 3.4
pair_modify shift yes

mass 1 39.948

timestep 0.001

velocity all create 0.0 ${ibead}

fix 1 all pimd/langevin method pimd ensemble nvt integrator obabo thermostat PILE_L 1234 tau 1.0 temp 113.15 taup 1.0 fixcom no

thermo_style custom step temp f_1[*] vol press
thermo 100
thermo_modify norm no

# dump dcd all custom 100 ${ibead}.dcd id type xu yu zu vx vy vz ix iy iz fx fy fz
# dump_modify dcd sort id format line "%d %d %.16f %.16f %.16f %.16f %.16f %.16f %d %d %d %.16f %.16f %.16f"

run 1000
