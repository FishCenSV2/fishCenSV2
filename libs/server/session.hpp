#pragma once

#include <boost/asio.hpp>
#include <mutex>
#include <cstdint>
#include "server.hpp"

using boost::asio::ip::tcp;

/**
*  
* @brief An class that represents a session between the client and
*        server. It runs based on an asynchronous callback model.
*
* A class which represents an active session between a client and
* the server where the client sends a request and the server sends
* back data containing the counts for each species.
*
* NOTE: The client only has to send any number for them to receive data.
*       This will remain like this until other values like temperature
*       is implemented.
*
* @author Tristan Huen
*/
class Session : public std::enable_shared_from_this<Session> {
public:
	std::vector<int> data_cpy;  ///< Copy of data to send to clients.
	unsigned request;		    ///< A request from the client to send data.
	Server& server;			    ///< Reference to server object for copying image data.

	/** @brief Constructor for Session.
	*
	* @param socket A socket object (note object is moved).
	* @param server The server object creating the session.
	* @return None
	*/
	Session(tcp::socket socket, Server& server);

	/** @brief Starts the session by calling do_read()
	* 
	* @return None
	*/
	void start();

private:

	tcp::socket _socket; ///< Socket object

	/** @brief Reads data from the client
	*
	* @return None
	*/
	void do_read();

	/** @brief Writes counting data to client
	*
	* @return None
	*/
	void do_write();

};