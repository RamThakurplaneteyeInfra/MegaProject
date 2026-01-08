# Village Boundary API

A FastAPI-based web service for managing and querying village boundaries in India. This application allows uploading KML files containing village and district boundary data, storing them in a PostgreSQL database with PostGIS extensions, and providing RESTful endpoints to retrieve geospatial information.

## Features

- **KML Upload**: Upload KML files to import village and district boundaries.
- **Geospatial Queries**: Retrieve districts, sub-districts, and villages with their geometric boundaries.
- **FastAPI Framework**: Built with FastAPI for high-performance asynchronous APIs.
- **PostgreSQL + PostGIS**: Robust geospatial database support.
- **GeoJSON Output**: API responses include GeoJSON-formatted coordinates for easy integration with mapping libraries.

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL with PostGIS extension
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

3. Set up the database:
   - Create a PostgreSQL database with PostGIS extension enabled.
   - Update the database connection string in `db.py` if necessary.

4. Run database migrations (if any):
   - The application uses SQLAlchemy to create tables automatically on startup.

## Running the Application

Start the FastAPI server using uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 9099 --reload
```

The API will be available at `http://localhost:9099`.

You can also access the interactive API documentation at `http://localhost:9099/docs`.

## API Endpoints

### Upload Endpoints

- `POST /upload-kml`: Upload a KML file containing village boundaries.
- `POST /upload-district-kml`: Upload a KML file containing district boundaries.

### Query Endpoints

- `GET /Alldistricts`: Retrieve all districts with their boundaries.
- `GET /sub-districts?district=<district_name>`: Get sub-districts for a specific district.
- `GET /villages-by-subdistrict?subdistrict=<subdistrict_name>`: Get villages for a specific sub-district with geometries.
- `GET /districts`: List all unique district names.
- `GET /subdistricts`: List all unique sub-district names.
- `GET /village-by-subdist?sub_dist=<sub_dist>&village_name=<village_name>`: Get a specific village by sub-district and name.
- `GET /districts/geo?district=<district_name>`: Get villages within a district with coordinates.
- `GET /subdistricts/by-district?district=<district_name>`: Get sub-districts for a district.
- `GET /subdistricts/by-district/geo?district=<district_name>`: Get sub-districts with coordinates.
- `GET /villages`: List all unique village names.
- `GET /villages/by-district?district=<district_name>`: Get villages for a district.
- `GET /villages/by-district/geo?district=<district_name>`: Get villages with coordinates.
- `GET /villages/by-subdist?sub_dist=<sub_dist>`: Get villages for a sub-district.
- `GET /villages/by-subdist/geo?sub_dist=<sub_dist>`: Get villages with coordinates.
- `GET /health`: Health check endpoint.

## Dependencies

- fastapi: Web framework for building APIs.
- uvicorn: ASGI server for running FastAPI.
- sqlalchemy: SQL toolkit and ORM.
- psycopg2-binary: PostgreSQL adapter for Python.
- geoalchemy2: Geospatial extension for SQLAlchemy.
- geopandas: Geospatial data manipulation.
- shapely: Geometric operations.
- fiona: Reading/writing geospatial data files.
- pyproj: Cartographic projections and coordinate transformations.
- python-dotenv: Environment variable management.
- lxml: XML processing for KML parsing.
- python-multipart: Multipart form data handling.

## Database Schema

The application uses three main tables:

- `districts`: Stores district information with polygon geometries.
- `village`: Stores village metadata (name, district, sub-district, state).
- `village_boundaries`: Stores village polygon geometries linked to villages.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Make your changes and commit: `git commit -am 'Add feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
