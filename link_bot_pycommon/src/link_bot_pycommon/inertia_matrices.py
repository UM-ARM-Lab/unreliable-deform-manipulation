def sphere(mass, radius):
    return [
        2.0 / 3 * mass * radius ** 2,
        2.0 / 3 * mass * radius ** 2,
        2.0 / 3 * mass * radius ** 2,
    ]


def cylinder(mass, radius, length):
    return [
        1.0 / 12 * mass * (3 * radius ** 2 + length ** 2),
        1.0 / 12 * mass * (3 * radius ** 2 + length ** 2),
        1.0 / 2 * mass * radius ** 2,
    ]


def box(mass, x, y, z):
    return [
        1.0 / 12 * mass * (z ** 2 + y ** 2),
        1.0 / 12 * mass * (x ** 2 + z ** 2),
        1.0 / 12 * mass * (x ** 2 + y ** 2),
    ]
