from sqlalchemy import Column, BigInteger, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry

Base = declarative_base()


class District(Base):
    __tablename__ = "districts"

    id = Column(BigInteger, primary_key=True, index=True)
    district_name = Column(Text, unique=True, nullable=False)
    state = Column(Text)
    geom = Column(Geometry("POLYGON", srid=4326))


class Village(Base):
    __tablename__ = "village"

    id = Column(BigInteger, primary_key=True, index=True)
    village_name = Column(Text, nullable=False)
    district = Column(Text, nullable=False)
    sub_dist = Column(Text)
    state = Column(Text)


class VillageBoundary(Base):
    __tablename__ = "village_boundaries"

    id = Column(BigInteger, primary_key=True, index=True)
    village_id = Column(
        BigInteger,
        ForeignKey("village.id", ondelete="CASCADE"),
        nullable=False
    )
    geom = Column(Geometry("POLYGON", srid=4326), nullable=False)
